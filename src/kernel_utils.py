import os

from dotenv import load_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.core_plugins.time_plugin import TimePlugin

from src.myskills import WeatherPlugin, MathPlugin, InternetSearchPlugin
from src.prompt_template import build_system_message

# Load environment variables from .env file
load_dotenv()

# Function to get active plugins based on configuration
def get_active_plugins(plugin_config):
    plugins = {}
    if plugin_config.get("TimePlugin"):
        print("Appended TimePlugin")
        plugins["TimePlugin"] = TimePlugin()
    
    # Weather plugin
    if plugin_config.get("WeatherPlugin"):
        print("Appended WeatherPlugin")
        plugins["WeatherPlugin"] = WeatherPlugin()

    # Math plugin
    if plugin_config.get("MathPlugin"):
        print("Appended MathPlugin")
        plugins["MathPlugin"] = MathPlugin()

    # Internet search plugin
    if plugin_config.get("InternetSearchPlugin"):
        print("Appended InternetSearchPlugin")
        plugins["InternetSearchPlugin"] = InternetSearchPlugin()

    return plugins


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

    # Create a chat history object to store conversation history
    chat_history = ChatHistory()
    # Build the system message from your 3 text areas
    system_msg = build_system_message(config["context_text"], config["filters_text"], config["output_format_text"])
    # Add the system message to the kernel's chat history
    chat_history.add_system_message(system_msg)
    
    # Dynamically add plugins to the kernel
    active_plugins = get_active_plugins(config["plugins"])
    for plugin_name, plugin_function in active_plugins.items():
        kernel.add_plugin(plugin_function, plugin_name=plugin_name)


    return kernel, chat_completion, chat_history

async def get_reply(config:dict, kernel: sk.Kernel, user_input: str, history: ChatHistory, chat_completion: AzureChatCompletion) -> str:
    """
    Asynchronously get a reply from the Semantic Kernel.
    This function creates execution settings, adds the user message to history,
    and awaits a response.
    """
    # Enable planning or auto-selection if needed.
    service_id = chat_completion.service_id  
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
    )
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
