import os
import tiktoken 
import asyncio
import re, pathlib, aiohttp

from bs4 import BeautifulSoup 
from typing import AsyncGenerator, Tuple
from typing import List, Callable, Optional
from dotenv import load_dotenv

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding 
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore 
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory  

# Load environment variables from .env file
load_dotenv()

# Load essential environment variables for Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

def init_memory() -> SemanticTextMemory:
    """
    Return a ready‑to‑use SemanticTextMemory object.
    Nothing is registered on the kernel; you just hold the handle.
    """
    embedding = AzureTextEmbedding(
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        deployment_name=os.getenv(
            "AZURE_OPENAI_EMBEDDEMENT",   # fallback to env var or hard‑code
            "text-embedding-ada-002",
        ),
    )
    store   = VolatileMemoryStore()
    memory  = SemanticTextMemory(store, embedding)
    return memory

def _split_into_chunks(text:str, max_tokens:int=500, overlap:int=100):
    enc = tiktoken.get_encoding("cl100k_base")  
    tokens = enc.encode(text)
    stride = max_tokens - overlap
    for i in range(0, len(tokens), stride):
        chunk = enc.decode(tokens[i : i + max_tokens])
        yield chunk

async def ingest_file( kernel: sk.Kernel, memory: SemanticTextMemory, file_name: str, text: str, collection: str = "uploaded_docs", max_concurrency: int = 15, on_progress: Optional[Callable[[float], None]] = None) -> None:
    """
    Break `text` into chunks, embed & store them **concurrently**.
    The optional `on_progress` is called with a 0‑1 fraction after
    each chunk completes.
    """
    chunks: List[str] = list(_split_into_chunks(text))
    print(f"Generated chunks → {len(chunks)} chunks")
    total = len(chunks)

    sem   = asyncio.Semaphore(max_concurrency)
    done  = 0                    # will be captured by the inner coroutine
    lock  = asyncio.Lock()       # to guard the counter when we update

    async def _save(idx: int, chunk: str):
        nonlocal done
        async with sem:          # throttle concurrent requests
            doc_id = f"{file_name}-{idx}"
            await memory.save_information(
                collection,
                id=doc_id,
                text=chunk,
                description=file_name,
            )
        # update the shared counter & fire progress callback
        async with lock:
            done += 1
            if on_progress:
                on_progress(done / total)

    # create and launch all the tasks up‑front
    tasks = [
        asyncio.create_task(_save(i, c))
        for i, c in enumerate(chunks, start=1)
    ]
    # Wait for *all* of them to finish (raises if any task errors)
    await asyncio.gather(*tasks)

async def retrieve_relevant_chunks(memory: SemanticTextMemory, query:str,
                                   collection:str="uploaded_docs",
                                   top_k:int=3, min_relevance:float=0.7):
    results = await memory.search(
        collection, query, limit=top_k, min_relevance_score=min_relevance
    )
    return [r.text for r in results]


# ── 1)  Generic webpage ──────────────────────────────────────
async def _fetch_url_text(url: str, session: aiohttp.ClientSession) -> str:
    async with session.get(url, timeout=30) as resp:
        html = await resp.text()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")

async def ingest_webpage(
    kernel: sk.Kernel,
    memory: SemanticTextMemory,
    url: str,
    collection: str = "uploaded_docs",
) -> None:
    """
    Downloads <url>, strips markup → plain text, then feeds it to ingest_file().
    """
    async with aiohttp.ClientSession() as sess:
        text = await _fetch_url_text(url, sess)
    await ingest_file(kernel, memory, url, text, collection)


# ── 2)  GitHub repository ────────────────────────────────────
_GH_RE = re.compile(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#]+)")

RAW_TPL = "https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
API_TPL = "https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"

_TEXT_EXT = {
    ".md", ".txt", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".go",
    ".rs", ".html", ".css", ".json", ".yml", ".yaml", ".toml",
}

async def _walk_github(
    owner: str,
    repo: str,
    branch: str,
    path: str,
    session: aiohttp.ClientSession,
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Yields (path_in_repo, raw_url) pairs for text‑like files.
    Falls back from 'main' to 'master' automatically.
    """
    async with session.get(API_TPL.format(owner=owner, repo=repo,
                                          path=path, branch=branch),
                           headers={"Accept": "application/vnd.github.v3+json"}) as resp:
        if resp.status == 404 and branch == "main":
            async for t in _walk_github(owner, repo, "master", path, session):
                yield t
            return
        data = await resp.json()

    # make sure we iterate a list
    if isinstance(data, dict) and data.get("type") == "file":
        data = [data]

    for entry in data:
        if entry["type"] == "dir":
            async for t in _walk_github(owner, repo, branch, entry["path"], session):
                yield t
        elif entry["type"] == "file":
            ext = pathlib.Path(entry["name"]).suffix.lower()
            if ext in _TEXT_EXT and entry["size"] < 200_000:
                raw_url = RAW_TPL.format(owner=owner, repo=repo,
                                         branch=branch, path=entry["path"])
                yield entry["path"], raw_url

async def ingest_github_repo(
    kernel: sk.Kernel,
    memory: SemanticTextMemory,
    repo_url: str,
    collection: str = "uploaded_docs",
) -> None:
    """
    Crawls a public GitHub repo and indexes every text/code file.
    """
    m = _GH_RE.match(repo_url)
    if not m:
        raise ValueError("Not a valid GitHub repository URL")

    owner, repo = m.group("owner"), m.group("repo")
    async with aiohttp.ClientSession() as sess:
        async for rel_path, raw_url in _walk_github(owner, repo, "main", "", sess):
            async with sess.get(raw_url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    doc_id = f"{owner}/{repo}:{rel_path}"
                    await ingest_file(kernel, memory, doc_id, text, collection)