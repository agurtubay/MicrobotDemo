# prompt_template.py
def build_system_message(context_text: str, filters_text: str, output_format_text: str) -> str:
    """
    Builds a system instruction containing context, filters, 
    and output format guidelines.
    """
    system_msg = f"""CONTEXT:
                    {context_text}

                    When a chart is useful, CALL exactly one of these tools with the correct arguments:

                    - bar_chart(categories, values, title?, x_label?, y_label?)
                    - line_chart(x, y, title?, x_label?, y_label?)
                    - area_chart(x, y, title?, x_label?, y_label?)
                    - scatter_chart(x, y, title?, x_label?, y_label?)
                    - hist_chart(values, bins?, title?, x_label?, y_label?)
                    - box_chart(groups, group_labels?, title?, y_label?)

                    Take the PNG data URL the tool returns and append it to answer.images[].
                    Also include a short textual insight in answer.text.

                    IMPORTANT: When you generate a chart, CALL exactly one chart tool (bar_chart, line_chart, area_chart, scatter_chart, hist_chart, box_chart).
                    The tool returns a RELATIVE PNG file path like "generated_charts/bar_abc123.png".
                    Append that path string to answer.images[].
                    Do NOT return absolute Windows paths, and do NOT return base64 or Markdown images.
                    Your final message MUST be a single JSON object following the ResponseEnvelope schema.

                    FILTERS:
                    {filters_text}

                    OUTPUT_FORMAT:
                    {output_format_text}
                """
    return system_msg