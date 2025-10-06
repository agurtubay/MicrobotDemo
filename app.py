# app.py — MicroBot (simplified & fixed)
# --------------------------------------
# Key changes:
# - st.set_page_config is the FIRST Streamlit call (fixes SetPageConfig error).
# - Smaller, readable helpers (render_envelope, normalize_reply_to_envelope, etc.).
# - Robust local image resolver + compact image width.
# - Clean message renderer with styled bubbles that still support Markdown.
# - Same features you already had (RAG uploads, charts, tables, images).

import asyncio
import json, base64, io, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import markdown as md
from jsonschema import validate
import streamlit as st
from pypdf import PdfReader

from semantic_kernel.contents.chat_history import ChatHistory
from src.kernel_utils import initialize_kernel, get_reply
from src.indexing import (
    init_memory, ingest_file, retrieve_relevant_chunks,
    ingest_webpage, ingest_github_repo
)

# ---------------------------------------------------------------------
# 0) PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND
# ---------------------------------------------------------------------
st.set_page_config(page_title="MicroBot", layout="wide")

# ---------------------------------------------------------------------
# 1) CONSTANTS & CSS
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
IMG_WIDTH_PX = 520  # central, readable chart width

# Minimal, stable CSS (no brittle auto-generated class selectors)
st.markdown("""
<style>
/* App background */
html, body, [class^='stMain'] { background-color: #DCEAF7 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #156082 !important; }
[data-testid="stSidebar"] * { color: #ECECEC !important; }

/* Header centering */
#centered-header { text-align: center; margin: 16px 0 8px 0; }

/* Bubbles */
.user-bubble{
  background:#2E8B57; color:#fff; border-radius:10px; padding:8px 12px;
  display:inline-block; max-width:80%;
}
.assistant-title{
  background:#333; color:#fff; border-radius:10px; padding:6px 10px;
  display:inline-block; margin:5px 0;
}
.assistant-bubble{
  background:#333; color:#fff; border-radius:10px; padding:10px 12px;
  display:inline-block; max-width:80%; line-height:1.35;
}
.assistant-bubble a{ color:#cfe8ff; text-decoration:underline; }
.assistant-bubble code{ background:#222; padding:2px 4px; border-radius:4px; }
.assistant-bubble pre{ background:#222; padding:8px; border-radius:6px; overflow:auto; }
.assistant-bubble img{ max-width:100%; height:auto; border-radius:8px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ResponseEnvelope schema (kept compact)
RESPONSE_ENVELOPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ok", "insufficient_data", "error"]},
        "answer": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "text": {"type": "string"},
                "table": {
                    "type": "object",
                    "properties": {
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {"type": "array", "items": {"type": "array"}}
                    },
                    "required": ["columns", "rows"]
                },
                "chart": {  # optional UI-rendered charts
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["bar","line","area","scatter","box","hist"]},
                        "title": {"type": "string"},
                        "x_label": {"type": "string"},
                        "y_label": {"type": "string"},
                        "x": {"type": "array"},
                        "y": {"type": "array"},
                        "series": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "x": {"type": "array"},
                                "y": {"type": "array"}
                            },
                            "required": ["x", "y"]
                        }},
                        "bins": {"type": ["integer", "null"]},
                        "groups": {"type": "array", "items": {"type": "array"}},
                        "group_labels": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "images": {"type": "array", "items": {"type": "string"}}
            }
        },
        "citations": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["status", "answer"]
}

# ---------------------------------------------------------------------
# 2) HELPERS
# ---------------------------------------------------------------------
def ensure_memory():
    if "memory" not in st.session_state:
        st.session_state["memory"] = init_memory()

def _render_chart(spec: dict):
    """UI-side chart renderer (works when assistant fills answer.chart)."""
    kind = (spec.get("type") or "bar").lower()
    title = spec.get("title", "")
    xlab = spec.get("x_label", "")
    ylab = spec.get("y_label", "")
    fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")

    def to_np(v):
        try: return np.array(v)
        except Exception: return np.array([])

    series = spec.get("series")
    if series:
        if kind in ("bar","column"):
            n = len(series)
            x = to_np(series[0].get("x", []))
            idx = np.arange(len(x))
            width = 0.8 / max(n, 1)
            for i, s in enumerate(series):
                y = to_np(s.get("y", []))
                ax.bar(idx + i*width, y, width=width, label=s.get("name", f"series{i+1}"))
            ax.set_xticks(idx + width*(n-1)/2)
            ax.set_xticklabels(x)
            ax.legend()
        elif kind in ("line","area"):
            for i, s in enumerate(series):
                x = to_np(s.get("x", [])); y = to_np(s.get("y", []))
                ax.plot(x, y, label=s.get("name", f"series{i+1}"))
                if kind == "area": ax.fill_between(x, y, alpha=0.2)
            ax.legend()
        elif kind in ("scatter","dots"):
            for i, s in enumerate(series):
                x = to_np(s.get("x", [])); y = to_np(s.get("y", []))
                ax.scatter(x, y, label=s.get("name", f"series{i+1}"))
            ax.legend()
        else:
            s0 = series[0]
            ax.plot(to_np(s0.get("x", [])), to_np(s0.get("y", [])))
    else:
        x = to_np(spec.get("x", [])); y = to_np(spec.get("y", []))
        if kind in ("bar","column"): ax.bar(x, y)
        elif kind == "line": ax.plot(x, y)
        elif kind == "area": ax.plot(x, y); ax.fill_between(x, y, alpha=0.2)
        elif kind in ("scatter","dots"): ax.scatter(x, y)
        elif kind in ("hist","histogram"): ax.hist(y if y.size else x, bins=spec.get("bins") or 10)
        elif kind in ("box","boxplot"):
            groups = spec.get("groups"); labels = spec.get("group_labels")
            if groups: ax.boxplot(groups, labels=labels if labels else None, vert=True)
            else: ax.boxplot(y if y.size else x, vert=True)
        else:
            ax.plot(x, y)

    if title: ax.set_title(title)
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def _show_local_image(path_str: str, width: int = IMG_WIDTH_PX):
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p  # keep inside project
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    st.image(str(p), width=width)

def render_envelope(env: dict):
    ans = env.get("answer", {}) or {}

    # 1) text
    if ans.get("text"):
        body_html = md.markdown(ans["text"], extensions=["tables", "fenced_code"])
        st.markdown(f'<div class="assistant-bubble">{body_html}</div>', unsafe_allow_html=True)

    # 2) table
    if ans.get("table"):
        cols = ans["table"].get("columns", [])
        rows = ans["table"].get("rows", [])
        try: st.table(pd.DataFrame(rows, columns=cols))
        except Exception: st.write({"columns": cols, "rows": rows})

    # 3) chart (UI-rendered)
    if ans.get("chart"):
        _render_chart(ans["chart"])

    # 4) images (paths, http(s), or data-url)
    imgs = ans.get("images") or []
    for img in imgs:
        try:
            if isinstance(img, str) and img.startswith("data:image"):
                b64 = img.split(",", 1)[1]
                pad = (-len(b64)) % 4
                if pad: b64 += "=" * pad
                st.image(io.BytesIO(base64.b64decode(b64)), width=IMG_WIDTH_PX)
            elif isinstance(img, str):
                if img.lower().startswith(("http://", "https://")):
                    st.image(img, width=IMG_WIDTH_PX)
                else:
                    _show_local_image(img, width=IMG_WIDTH_PX)
            else:
                st.warning(f"Unsupported image entry: {type(img)}")
        except Exception as e:
            st.error(f"Image render failed: {type(e).__name__}: {e}")

    # 5) citations
    cits = env.get("citations") or []
    if cits:
        with st.expander("Citations"):
            for c in cits: st.write(c)

# Envelope normalization (simple & resilient)
DATA_URL_PATTERN = re.compile(
    r'!\[[^\]]*\]\(\s*(data:image\/[^\)]+)\s*\)|(?P<url>data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+)',
    flags=re.IGNORECASE
)

def _try_json(s: str):
    try: return json.loads(s)
    except Exception: return None

def _is_envelope(obj: dict) -> bool:
    return isinstance(obj, dict) and "status" in obj and "answer" in obj

def _wrap_near_envelope(obj: dict):
    if not isinstance(obj, dict): return None
    answer_keys = ("text", "table", "chart", "images")
    answer = {k: obj[k] for k in answer_keys if k in obj}
    if not answer: return None
    return {"status": obj.get("status", "ok"), "answer": answer, "citations": obj.get("citations", [])}

def _coerce_text_to_envelope(text: str):
    imgs = []
    for m in DATA_URL_PATTERN.finditer(text):
        url = m.group(1) or m.group("url")
        if url: imgs.append(url)
    cleaned = re.sub(r'!\[[^\]]*\]\(\s*data:image\/[^\)]+\s*\)', '', text, flags=re.IGNORECASE)
    for u in imgs: cleaned = cleaned.replace(u, '')
    cleaned = cleaned.strip()
    if imgs:
        return {"status": "ok", "answer": {"text": cleaned or " ", "images": imgs}, "citations": []}
    return None

def _validate_envelope(env: dict) -> bool:
    try:
        validate(instance=env, schema=RESPONSE_ENVELOPE_SCHEMA)
        return True
    except Exception:
        return False

def normalize_reply_to_envelope(reply_text: str):
    obj = _try_json(reply_text)
    if isinstance(obj, dict):
        if _is_envelope(obj) and _validate_envelope(obj): return obj
        wrapped = _wrap_near_envelope(obj)
        if wrapped and _validate_envelope(wrapped): return wrapped
    coerced = _coerce_text_to_envelope(reply_text)
    if coerced and _validate_envelope(coerced): return coerced
    return None

# ---------------------------------------------------------------------
# 3) SESSION DEFAULTS
# ---------------------------------------------------------------------
if "kernel_config" not in st.session_state:
    st.session_state["kernel_config"] = {
        "selected_model": "gpt-4o-mini",
        "max_tokens": 100,
        "temperature": 0.7,
        "context_text": "You are a general purpose assistant.",
        "filters_text": "",
        "output_format_text": "",
        "plugins": {}
    }

if "kernel" not in st.session_state or "chat_completion" not in st.session_state:
    st.session_state["kernel"], st.session_state["chat_completion"], st.session_state["chat_history"] = initialize_kernel(
        st.session_state["kernel_config"]
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# ---------------------------------------------------------------------
# 4) HEADER
# ---------------------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([0.2, 0.6, 0.2])
with header_col2:
    st.image("assets/assistant_logo.png", width=200)
    st.markdown("""
    <div id="centered-header">
        <h2 style="margin:0;color:#156082;">MicroBot</h2>
        <h3 style="margin:0;color:#156082;">Microsoft’s Demo AI Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    st.image("assets/microsoft_logo.png", width=160)

# ---------------------------------------------------------------------
# 5) SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    view = st.radio("Website Tabs", ["💬 Chat", "📂 Upload & Index"])
    st.markdown("---")

    st.markdown("### Model Parameters")
    max_tokens = st.slider("Max Output Tokens", 10, 2000, st.session_state["kernel_config"]["max_tokens"], 10)
    temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state["kernel_config"]["temperature"]), 0.1)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-5-mini"], index=0)

    st.markdown("---")
    st.markdown("## Prompt Template Configuration")
    context_text = st.text_area("Context", value=st.session_state["kernel_config"].get("context_text", ""))
    filters_text = st.text_area("Filters", value=st.session_state["kernel_config"].get("filters_text", ""))
    output_format_text = st.text_area("Output Format", value=st.session_state["kernel_config"].get("output_format_text", ""))

    st.markdown("---")
    st.markdown("### Plugins")
    weather_plugin   = st.checkbox("Weather", value=False)
    time_plugin      = st.checkbox("Time", value=False)
    math_plugin      = st.checkbox("Math", value=False)
    internet_plugin  = st.checkbox("Internet Search", value=False)
    chart_plugin     = st.checkbox("Charts", value=False)

    if st.button("Apply Configuration"):
        cfg = {
            "selected_model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "context_text": context_text,
            "filters_text": filters_text,
            "output_format_text": output_format_text,
            "plugins": {
                "TimePlugin": time_plugin or None,
                "WeatherPlugin": weather_plugin or None,
                "MathPlugin": math_plugin or None,
                "InternetSearchPlugin": internet_plugin or None,
                "ChartsPlugin": chart_plugin or None,
            },
        }
        st.session_state["kernel_config"] = cfg
        st.session_state["kernel"], st.session_state["chat_completion"], st.session_state["chat_history"] = initialize_kernel(cfg)
        st.session_state["messages"] = []
        st.success("Configuration applied. Chat reset.")

# ---------------------------------------------------------------------
# 6) VIEWS
# ---------------------------------------------------------------------
if view == "💬 Chat":
    # Render chat history
    for msg in st.session_state["messages"]:
        role = msg.get("role", "")
        if role == "user":
            st.markdown(f'<div class="user-bubble"><strong>You:</strong> {msg.get("content","")}</div>',
                        unsafe_allow_html=True)
        elif role == "assistant_structured":
            st.markdown('<div class="assistant-title"><strong>Assistant:</strong></div>', unsafe_allow_html=True)
            render_envelope(msg.get("envelope", {}))
            with st.expander("Show raw JSON"):
                st.code(json.dumps(msg.get("envelope", {}), indent=2))
        else:  # plain assistant
            st.markdown('<div class="assistant-title"><strong>Assistant:</strong></div>', unsafe_allow_html=True)
            body = msg.get("content", "")
            body_html = md.markdown(body, extensions=["tables", "fenced_code"])
            st.markdown(f'<div class="assistant-bubble">{body_html}</div>', unsafe_allow_html=True)

    # Chat input (pinned at bottom)
    user_input = st.chat_input("Ask me anything…")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Prepare kernel and prompt
        kernel = st.session_state["kernel"]
        chat_completion = st.session_state["chat_completion"]
        chat_history: ChatHistory = st.session_state["chat_history"]
        config = st.session_state["kernel_config"]

        # RAG (if memory exists)
        if "memory" in st.session_state:
            rel = asyncio.run(retrieve_relevant_chunks(st.session_state["memory"], user_input, top_k=3))
            rag_context = "\n".join(f"• {c}" for c in rel) if rel else ""
            final_user_input = f"Refer to the following knowledge if helpful:\n{rag_context}\n\nUser question: {user_input}"
        else:
            final_user_input = user_input

        # Get assistant reply
        try:
            reply_text, chat_history = asyncio.run(get_reply(config, kernel, final_user_input, chat_history, chat_completion))
        except Exception as e:
            reply_text = f"Kernel Error: {e}"

        # Normalize into envelope if possible
        safe_text = reply_text if isinstance(reply_text, str) else str(reply_text)
        env = normalize_reply_to_envelope(safe_text)
        if env:
            st.session_state["messages"].append({"role": "assistant_structured", "envelope": env})
        else:
            st.session_state["messages"].append({"role": "assistant", "content": safe_text})

        st.rerun()

else:  # 📂 Upload & Index
    ensure_memory()
    kernel = st.session_state["kernel"]
    memory = st.session_state["memory"]

    st.markdown("<h3>Add documents to your knowledge base</h3>", unsafe_allow_html=True)
    files = st.file_uploader("Drag-drop or Browse", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if files: st.session_state["uploaded_files"] = files

    if st.button("Index selected files") and st.session_state["uploaded_files"]:
        for f in st.session_state["uploaded_files"]:
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                reader = PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")

            status = st.empty()
            bar = st.progress(0.0)
            status.write(f"Embedding **{f.name}** …")

            def _update(frac): bar.progress(frac)

            asyncio.run(ingest_file(kernel, memory, f.name, text, on_progress=_update))
            bar.empty()
            status.success(f"Indexed {f.name} ✅")

        st.success("All files indexed! 🎉")
        st.session_state["uploaded_files"] = []

    st.markdown("---")
    st.markdown("<h3>Add webpages or a GitHub repo</h3>", unsafe_allow_html=True)
    url = st.text_input("URL or GitHub repo")
    c1, c2 = st.columns(2)
    if c1.button("Index webpage") and url:
        with st.spinner("Fetching & embedding webpage…"):
            asyncio.run(ingest_webpage(kernel, memory, url))
        st.success("Webpage indexed!")
    if c2.button("Index GitHub repo") and url:
        with st.spinner("Crawling & embedding repo…"):
            asyncio.run(ingest_github_repo(kernel, memory, url))
        st.success("Repository indexed!")
