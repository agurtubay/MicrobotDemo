def get_config_from_ui(selected_model: str, max_tokens: int, temperature: float, plugins: dict) -> dict:
    """
    Collect configuration settings from the UI and return a configuration dict.
    """
    return {
        "selected_model": selected_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "plugins": plugins
    }
