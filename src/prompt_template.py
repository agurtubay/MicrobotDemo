# prompt_template.py
def build_system_message(context_text: str, filters_text: str, output_format_text: str) -> str:
    """
    Builds a system instruction containing context, filters, 
    and output format guidelines.
    """
    system_msg = f"""CONTEXT:
                    {context_text}

                    FILTERS:
                    {filters_text}

                    OUTPUT_FORMAT:
                    {output_format_text}
                """
    return system_msg