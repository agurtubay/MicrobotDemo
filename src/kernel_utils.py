import os
import asyncio

from dotenv import load_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
# from src.plugins import get_active_plugins   <-- Plugins are ignored for now

def initialize_kernel(config: dict) -> sk.Kernel:
    """
    Initialize the Semantic Kernel based on the provided configuration.
    This includes setting the chosen model, max tokens, temperature,
    and dynamically adding plugins as needed.
    """
    # Create a new Kernel instance
    kernel = sk.Kernel()

    # Load environment variables from .env file
    load_dotenv()

    # Load essential environment variables for Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    print(f"Azure OpenAI API Key: {azure_api_key}")
    print(f"Azure OpenAI Endpoint: {azure_endpoint}")
    # Here, we use config['selected_model'] to determine the deployment name.
    deployment_name = os.getenv(f"AZURE_OPENAI_CHAT_DEPLOYMENT_{config['selected_model'].upper().replace(' ', '_')}", "")

    chat_completion = AzureChatCompletion(
        service_id=config['selected_model'],
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        deployment_name=deployment_name,
    )
    
    # Add Azure OpenAI service to the kernel
    kernel.add_service(chat_completion)

    # Adjust kernel settings for max_tokens and temperature.
    settings = kernel.get_prompt_execution_settings_from_service_id(config['selected_model'])
    settings.max_tokens = config["max_tokens"]
    settings.temperature = config["temperature"]

    # Plugins are attached here if used; for now they can be ignored.
    # active_plugins = get_active_plugins(config["plugins"])
    # for plugin_name, plugin_function in active_plugins.items():
    #     kernel.add_plugin(plugin_name, plugin_function)

    return kernel, chat_completion

async def get_reply(kernel: sk.Kernel, user_input: str, history: ChatHistory, chat_completion: AzureChatCompletion) -> str:
    """
    Asynchronously get a reply from the Semantic Kernel.
    This function creates execution settings, adds the user message to history,
    and awaits a response.
    """
    # Enable planning or auto-selection if needed.
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    # Add the user's message to the chat history.
    history.add_user_message(user_input)
    
    # Call your kernel's function to get the reply. Replace the following call
    # with your actual Semantic Kernel API call. For example:
    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel
    )
    
    # Add the assistant's reply to chat history.
    history.add_message(result)
    
    return str(result), history
