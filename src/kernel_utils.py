import os
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from src.plugins import get_active_plugins

def initialize_kernel(config: dict) -> sk.Kernel:
    """
    Initialize the Semantic Kernel based on the provided configuration.
    This includes setting the chosen model, max tokens, temperature,
    and dynamically adding plugins as needed.
    """
    # Create a new Kernel instance
    kernel = sk.Kernel()

    # Load essential environment variables for Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

    # For simplicity, assume config['selected_model'] maps to a deployment name in .env
    deployment_name = os.getenv(f"AZURE_OPENAI_CHAT_DEPLOYMENT_{config['selected_model'].upper().replace(' ', '_')}", "")
    
    # Add Azure OpenAI service to the kernel
    kernel.add_service(
        AzureChatCompletion(
            service_id=config['selected_model'],
            api_key=azure_api_key,
            endpoint=azure_endpoint,
            deployment_name=deployment_name,
        )
    )

    # You might have a function to adjust kernel settings for max_tokens and temperature.
    settings = kernel.get_prompt_execution_settings_from_service_id(config['selected_model'])
    settings.max_tokens = config["max_tokens"]
    settings.temperature = config["temperature"]

    # Attach plugins if any are active using a helper from plugins.py
    active_plugins = get_active_plugins(config["plugins"])
    for plugin_name, plugin_function in active_plugins.items():
        kernel.add_plugin(plugin_name, plugin_function)

    return kernel
