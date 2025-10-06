# src/kernel_utils.py
import os, sys, json, time, logging, functools, inspect
from pathlib import Path
from typing import Tuple

# ── 1) Load .env EARLY (root .env, optional src/.env, or explicit path) ─────────
from dotenv import load_dotenv, find_dotenv
load_dotenv()

if not os.getenv("MBOT_TRACE"): print("Tip: set MBOT_TRACE=1 in .env to enable trace logging")

# ── 2) SK + app imports ─────────────────────────────────────────────────────────
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.core_plugins.time_plugin import TimePlugin

from src.myskills import WeatherPlugin, MathPlugin, InternetSearchPlugin
from src.prompt_template import build_system_message

# ── 3) Tiny JSON logger to stdout (opt-in via MBOT_TRACE=1) ─────────────────────
TRACE_ENABLED = os.getenv("MBOT_TRACE", "0").lower() in {"1", "true"}
TRACE_PRETTY  = os.getenv("MBOT_TRACE_PRETTY", "0").lower() in {"1", "true"}

_trace_logger = logging.getLogger("microbot.trace")
if not _trace_logger.handlers:
    _trace_logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    _trace_logger.addHandler(h)

def _trace(event: str, **payload):
    if not TRACE_ENABLED:
        return
    rec = {"ts": int(time.time() * 1000), "event": event, **payload}
    msg = json.dumps(rec, ensure_ascii=False, indent=2) if TRACE_PRETTY else json.dumps(rec, ensure_ascii=False)
    _trace_logger.info(msg)

# ── 4) Exec settings (with optional “force tools” for demos) ────────────────────
_FIXED_TEMP_TAGS = ("o", "reasoning", "gpt-5")
def _is_fixed_temp_model(model_name: str) -> bool:
    mn = (model_name or "").lower()
    return any(tag in mn for tag in _FIXED_TEMP_TAGS)

def _build_exec_settings(config: dict, model_name: str) -> AzureChatPromptExecutionSettings:
    s = AzureChatPromptExecutionSettings(service_id=model_name)
    max_tokens = int(config.get("max_tokens", 256))
    # compatibility across SK versions
    if hasattr(s, "max_output_tokens"): s.max_output_tokens = max_tokens
    elif hasattr(s, "max_completion_tokens"): s.max_completion_tokens = max_tokens
    else: s.max_tokens = max_tokens
    if not _is_fixed_temp_model(model_name) and hasattr(s, "temperature"):
        s.temperature = float(config.get("temperature", 1.0))
    # function calling behavior
    if os.getenv("MBOT_FORCE_TOOLS", "0").lower() in {"1", "true"}:
        s.function_choice_behavior = FunctionChoiceBehavior.Required()
    else:
        s.function_choice_behavior = FunctionChoiceBehavior.Auto()
    return s

# ── 5) Wrap plugin methods for trace, while preserving SK markers ──────────────
def wrap_plugin_for_tracing(plugin, plugin_name, trace_fn):
    # preserve attributes SK uses to recognize kernel functions
    PRESERVE = functools.WRAPPER_ASSIGNMENTS + ("kernel_function", "metadata")
    for attr in dir(plugin):
        if attr.startswith("_"): continue
        fn = getattr(plugin, attr)
        if not callable(fn): continue
        orig = fn
        if inspect.iscoroutinefunction(orig):
            @functools.wraps(orig, assigned=PRESERVE)
            async def async_wrapper(*args, __orig=orig, __name=attr, **kwargs):
                trace_fn("tool_call", plugin=plugin_name, function=__name, arguments=kwargs)
                out = await __orig(*args, **kwargs)
                prev = (out[:800] + "…") if isinstance(out, str) and len(out) > 800 else out
                trace_fn("tool_result", plugin=plugin_name, function=__name, result_preview=prev)
                return out
            setattr(plugin, attr, async_wrapper)
        else:
            @functools.wraps(orig, assigned=PRESERVE)
            def sync_wrapper(*args, __orig=orig, __name=attr, **kwargs):
                trace_fn("tool_call", plugin=plugin_name, function=__name, arguments=kwargs)
                out = __orig(*args, **kwargs)
                prev = (out[:800] + "…") if isinstance(out, str) and len(out) > 800 else out
                trace_fn("tool_result", plugin=plugin_name, function=__name, result_preview=prev)
                return out
            setattr(plugin, attr, sync_wrapper)
    return plugin

# ── 6) Active plugins from toggles / env ───────────────────────────────────────
def get_active_plugins(plugin_config: dict) -> dict:
    plugins = {}
    if plugin_config.get("TimePlugin"):
        print("Appended TimePlugin");  plugins["TimePlugin"] = TimePlugin()
    if plugin_config.get("WeatherPlugin"):
        print("Appended WeatherPlugin"); plugins["WeatherPlugin"] = WeatherPlugin()
    if plugin_config.get("MathPlugin"):
        print("Appended MathPlugin"); plugins["MathPlugin"] = MathPlugin()
    if plugin_config.get("InternetSearchPlugin") and os.getenv("BRAVE_API_KEY"):
        print("Appended InternetSearchPlugin"); plugins["InternetSearchPlugin"] = InternetSearchPlugin()
    return plugins  # :contentReference[oaicite:0]{index=0}

# ── 7) Dump the EXACT OpenAI-style tools SK will advertise (SK 1.28 safe) ─────
def dump_advertised_tools_openai(kernel) -> list[dict]:
    metas = []
    plugins = getattr(kernel, "plugins", {}) or {}
    for _, kp in plugins.items():
        if hasattr(kp, "get_functions_metadata"):
            for m in kp.get_functions_metadata():
                if getattr(m, "is_prompt", False):
                    continue
                metas.append(m)

    def _json_type(t: str) -> str:
        t = (t or "").lower()
        if t in {"string", "str", "text"}: return "string"
        if t in {"int", "integer"}: return "integer"
        if t in {"float", "double", "number"}: return "number"
        if t in {"bool", "boolean"}: return "boolean"
        if t in {"array", "list", "tuple"}: return "array"
        if t in {"object", "dict", "map"}: return "object"
        return "string"

    tools = []
    for m in metas:
        props, required = {}, []
        for p in (getattr(m, "parameters", None) or []):
            prop = {"type": _json_type(getattr(p, "type", None))}
            if getattr(p, "description", None):
                prop["description"] = p.description
            if getattr(p, "is_required", False) or getattr(p, "default_value", None) is None:
                required.append(p.name)
            props[p.name] = prop

        tools.append({
            "type": "function",
            "function": {
                "name": f"{m.plugin_name}.{m.name}",
                "description": getattr(m, "description", None) or f"{m.plugin_name}.{m.name}",
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    tools.sort(key=lambda t: t["function"]["name"])
    return tools

# ── 8) Simple planner: uses the tool names from the advertised schema ─────────
async def planner_step(chat_completion, allowed_tool_names: list[str], user_input: str):
    plan_history = ChatHistory()
    plan_history.add_system_message(
        "You are a planning assistant. Given the user's query and the allowed tools, "
        "decide whether to call a tool. Output STRICT JSON with keys: "
        '{"intent": str, "plan": [str], "candidate_tools": [str], "chosen_tool": str|null, "arguments": object}. '
        "If no tool is needed, set chosen_tool to null and arguments to {}."
    )
    plan_history.add_system_message("Allowed tools:\n" + "\n".join(allowed_tool_names))
    plan_history.add_user_message(user_input)

    settings = AzureChatPromptExecutionSettings(temperature=0, max_tokens=350)
    res = await chat_completion.get_chat_message_content(chat_history=plan_history, settings=settings)
    text = getattr(res, "content", None) or (str(res[0].content) if hasattr(res, "__getitem__") else str(res))
    try:
        plan = json.loads(text)
    except Exception:
        plan = {"_raw": text}
    _trace("planner", allowed_tools=allowed_tool_names, plan=plan)
    return plan

# ── 9) Kernel init ─────────────────────────────────────────────────────────────
def initialize_kernel(config: dict) -> Tuple[sk.Kernel, AzureChatCompletion, ChatHistory]:
    kernel = sk.Kernel()

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_api_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not azure_endpoint or not azure_api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")
    if "/openai/" in azure_endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must be the base URL, e.g. https://<resource>.openai.azure.com")

    env_map = {
        "gpt-4o-mini": "AZURE_OPENAI_DEPLOYMENT_GPT_4O_MINI",
        "gpt-5-mini":  "AZURE_OPENAI_DEPLOYMENT_GPT_5_MINI",
    }
    selected = config.get("selected_model", "gpt-4o-mini")
    env_key  = env_map.get(selected)
    deployment_name = os.getenv(env_key, "") if env_key else ""
    if not deployment_name:
        raise RuntimeError(
            f"Missing deployment name for selected model '{selected}'. "
            f"Set {env_map.get(selected, '<DEPLOY_ENV_VAR>')} in your .env."
        )

    chat_completion = AzureChatCompletion(
        service_id=selected,
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        deployment_name=deployment_name,
        api_version=api_version,
    )
    kernel.add_service(chat_completion)

    chat_history = ChatHistory()
    system_msg = build_system_message(
        config.get("context_text", ""),
        config.get("filters_text", ""),
        config.get("output_format_text", ""),
    )
    chat_history.add_system_message(system_msg)

    # register only enabled plugins (wrapped)
    active_plugins = get_active_plugins(config.get("plugins", {}))
    for plugin_name, plugin in active_plugins.items():
        kernel.add_plugin(wrap_plugin_for_tracing(plugin, plugin_name, _trace), plugin_name=plugin_name)

    return kernel, chat_completion, chat_history

# ── 10) Chat helpers ───────────────────────────────────────────────────────────
def _message_to_text(msg) -> str:
    try:
        if getattr(msg, "content", None):
            return str(msg.content)
        items = getattr(msg, "items", None)
        if items:
            parts = [str(getattr(it, "text", "")) for it in items if getattr(it, "text", None)]
            if parts: return "".join(parts)
    except Exception:
        pass
    return str(msg)

# ── 11) Main turn with terminal traces (clean) ─────────────────────────────────
async def get_reply(config, kernel, user_input, history, chat_completion):
    model_name = chat_completion.service_id
    settings   = _build_exec_settings(config, model_name)

    # (a) dump the real tools payload (what the model sees)
    tools = dump_advertised_tools_openai(kernel)
    _trace("tools_schema_advertised", tools=tools)

    # (b) planner uses canonical tool names from the tools payload
    allowed_tool_names = [t["function"]["name"] for t in tools]
    try:
        await planner_step(chat_completion, allowed_tool_names, user_input)
    except Exception as e:
        _trace("planner_error", error=str(e))

    # (c) normal chat, with auto/required tool use as per settings
    history.add_user_message(user_input)
    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )

    # (d) surface any SK-attached tool items (not always present in 1.28)
    items = getattr(result, "items", []) or []
    for it in items:
        ev = {"_item_type": type(it).__name__}
        if hasattr(it, "plugin_name"): ev["plugin_name"] = it.plugin_name
        if hasattr(it, "name"):        ev["function"]    = it.name
        if hasattr(it, "arguments"):
            try: ev["arguments"] = json.loads(it.arguments)
            except Exception: ev["arguments"] = str(it.arguments)
        if hasattr(it, "result"):
            r = it.result
            ev["result_preview"] = (r[:800] + "…") if isinstance(r, str) and len(r) > 800 else r
        _trace("tool_event", **ev)

    # (e) final assistant text
    text = _message_to_text(result)
    history.add_assistant_message(text)
    _trace("assistant_message", text_preview=(text[:1000] + "…") if len(text) > 1000 else text)
    return text, history
