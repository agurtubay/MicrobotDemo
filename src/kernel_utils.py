import os
from typing import Tuple

from dotenv import load_dotenv, find_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from azure.core.exceptions import HttpResponseError

from src.myskills import WeatherPlugin, MathPlugin, InternetSearchPlugin
from src.prompt_template import build_system_message

# -----------------------------------------------------------------------------
# Robust .env loading (works with Streamlit changing CWD)
# -----------------------------------------------------------------------------
_dotenv_path = find_dotenv(filename="src/.env", usecwd=True)
if _dotenv_path:
    load_dotenv()
else:
    print("Warning: .env file not found – proceeding with process env only")


# Models that enforce fixed temperature (Azure reasoning / o-series)
_FIXED_TEMP_TAGS = ("o", "reasoning", "gpt-5")   


def _is_fixed_temp_model(model_name: str) -> bool:
    mn = (model_name or "").lower()
    return any(tag in mn for tag in _FIXED_TEMP_TAGS)


def _build_exec_settings(config: dict, model_name: str) -> AzureChatPromptExecutionSettings:
    """Create execution settings compatible with both legacy and new SDK fields.

    - Some models reject `max_tokens` and require `max_completion_tokens` or `max_output_tokens`.
    - Reasoning / o-series models often have fixed temperature = 1.
    """
    s = AzureChatPromptExecutionSettings(service_id=model_name)

    # ---- tokens field compatibility ----------------------------------------
    max_tokens = int(config.get("max_tokens", 256))
    if hasattr(s, "max_output_tokens"):
        s.max_output_tokens = max_tokens
    elif hasattr(s, "max_completion_tokens"):
        s.max_completion_tokens = max_tokens
    else:
        # legacy
        s.max_tokens = max_tokens

    # ---- temperature (skip for fixed-temp models) --------------------------
    if not _is_fixed_temp_model(model_name) and hasattr(s, "temperature"):
        s.temperature = float(config.get("temperature", 1.0))

    # ---- function calling behavior ----------------------------------------
    s.function_choice_behavior = FunctionChoiceBehavior.Auto()
    return s


# -----------------------------------------------------------------------------
# Plugins
# -----------------------------------------------------------------------------

def get_active_plugins(plugin_config: dict) -> dict:
    """Return instantiated plugins based on sidebar configuration.

    InternetSearchPlugin is only added if BRAVE_API_KEY is present to avoid
    raising ValueError during init.
    """
    plugins: dict = {}

    if plugin_config.get("TimePlugin"):
        print("Appended TimePlugin")
        plugins["TimePlugin"] = TimePlugin()

    if plugin_config.get("WeatherPlugin"):
        print("Appended WeatherPlugin")
        plugins["WeatherPlugin"] = WeatherPlugin()

    if plugin_config.get("MathPlugin"):
        print("Appended MathPlugin")
        plugins["MathPlugin"] = MathPlugin()

    if plugin_config.get("InternetSearchPlugin") and os.getenv("BRAVE_API_KEY"):
        print("Appended InternetSearchPlugin")
        plugins["InternetSearchPlugin"] = InternetSearchPlugin()

    return plugins


# -----------------------------------------------------------------------------
# Kernel + Chat service initialization
# -----------------------------------------------------------------------------

def initialize_kernel(config: dict) -> Tuple[sk.Kernel, AzureChatCompletion, ChatHistory]:
    """Initialize Semantic Kernel and Azure chat service.

    Returns (kernel, chat_completion, chat_history)
    """
    # Create a new Kernel instance
    kernel = sk.Kernel()

    # Azure base config
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not azure_endpoint or not azure_api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set.")
    if "/openai/" in azure_endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT must be the base URL, e.g. https://<resource>.openai.azure.com"
        )

    # Map UI model → env var with underscores (avoid hyphens in env var names)
    env_map = {
        "gpt-4o-mini": "AZURE_OPENAI_DEPLOYMENT_GPT_4O_MINI",
        "gpt-5-mini": "AZURE_OPENAI_DEPLOYMENT_GPT_5_MINI",
    }
    selected = config.get("selected_model", "gpt-4o-mini")
    env_key = env_map.get(selected)
    deployment_name = os.getenv(env_key, "") if env_key else ""

    if not deployment_name:
        debug_msg = (
            f"Selected model: {selected}"
            f"Env key looked up: {env_key}"
            f"AZURE_OPENAI_ENDPOINT present: {bool(azure_endpoint)}"
            f"AZURE_OPENAI_API_KEY present: {bool(azure_api_key)}"
        )
        raise RuntimeError("Missing deployment name for selected model." + debug_msg + "Set the corresponding env var in your .env to your Azure deployment name.")

    # Create chat completion service
    chat_completion = AzureChatCompletion(
        service_id=selected,
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        deployment_name=deployment_name,
        api_version=api_version,
    )

    # Register service with the kernel
    kernel.add_service(chat_completion)

    # System prompt + history
    chat_history = ChatHistory()
    system_msg = build_system_message(
        config.get("context_text", ""),
        config.get("filters_text", ""),
        config.get("output_format_text", ""),
    )
    chat_history.add_system_message(system_msg)

    # Dynamically add plugins
    active_plugins = get_active_plugins(config.get("plugins", {}))
    for plugin_name, plugin in active_plugins.items():
        kernel.add_plugin(plugin, plugin_name=plugin_name)

    return kernel, chat_completion, chat_history


# -----------------------------------------------------------------------------
# Chat loop
# -----------------------------------------------------------------------------
def _message_to_text(msg) -> str:
    # SK ChatMessageContent can hold plain .content or a list of items
    try:
        if getattr(msg, "content", None):
            return str(msg.content)
        items = getattr(msg, "items", None)
        if items:
            parts = []
            for it in items:
                t = getattr(it, "text", None)
                if t:
                    parts.append(str(t))
            if parts:
                return "".join(parts)
    except Exception:
        pass
    # Fallback
    return str(msg)

async def get_reply(
    config, kernel, user_input, history, chat_completion
):
    model_name = chat_completion.service_id
    settings = _build_exec_settings(config, model_name)

    history.add_user_message(user_input)

    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )

    text = _message_to_text(result)
    history.add_assistant_message(text)   # <- add plain text, not the raw object
    return text, history
