import asyncio

import streamlit as st
from pypdf import PdfReader
from semantic_kernel.contents.chat_history import ChatHistory

from src.kernel_utils import initialize_kernel, get_reply
from src.indexing import init_memory, ingest_file, retrieve_relevant_chunks,  ingest_webpage, ingest_github_repo

# ----- Custom CSS for dark theme and positioning -----
CUSTOM_CSS = """
<style>
/* Darken sidebar background */
[data-testid="stSidebar"] {
    background-color: #156082 !important;
}

/* Text color in main area to be lighter on dark background */
html, body, [class^='stMain']  {
    background-color: #DCEAF7 !important;
}

/* Force the widget labels (slider, checkbox) in the sidebar to be lighter text */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-cb, 
[data-testid="stSidebar"] .st-bb, [data-testid="stSidebar"] .st-bo {
    color: #ECECEC !important;
}

/* Sliders track background (darker tone) */
.css-q8sbsg.e1ii0kn30, .css-10trblm.e1ii0kn30 {
    background-color: #435366 !important;  /* Darker slider track */
}

# /* Position and style the 'Apply Configuration' button at bottom-right of sidebar */
# [class="stElementContainer element-container st-key-apply-config st-emotion-cache-kj6hex eu6p4el1"]{
#     position: fixed;
#     bottom: 20px;
#     color: #FFFFFF;
#     border-radius: 5px;
#     border: none;
#     padding: 0.6rem 1rem;
# }

/* Center the header (logo, title, subtitle) in main area */
#centered-header {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}
[data-testid="stFullScreenFrame"] {
    display: flex;
    justify-content: center;
}

/* Chat container: space to show messages above the fixed input bar */
#chat-container {
    padding-bottom: 80px; /* leave room for the fixed input bar */
}

[class = "stColumn st-emotion-cache-qvqwo6 eu6p4el2"] {
    display:flex;
    align-items:flex-end;
}

[class="st-emotion-cache-b0y9n5 em9zgd02"] {
    width:100%;
}

[class="st-emotion-cache-0 eu6p4el5"] {
    width:100%;
}

[class="st-emotion-cache-hzygls eht7o1d3"] {
    background-color: #DCEAF7;
}
</style>
"""
def ensure_memory():
    if "memory" not in st.session_state:
        st.session_state["memory"] = init_memory()


# -----------------------------------------------------------------------------
# DEFAULT KERNEL INITIALIZATION
# -----------------------------------------------------------------------------
# Define a default configuration. The key "selected_model" is used internally.
if "kernel_config" not in st.session_state:
    default_config = {
        "selected_model": "gpt-4o-mini",  # default model, adjust as needed
        "max_tokens": 100,                # default max tokens
        "temperature": 0.7,               # default temperature
        "context_text": "You are a general purpose assistant.",
        "filters_text": "",    
        "output_format_text": "",
        "plugins": {}                     # plugins will be ignored for now
    }
    st.session_state["kernel_config"] = default_config

# Retrieve or create the Kernel and ChatCompletion objects
if "kernel" not in st.session_state or "chat_completion" not in st.session_state:
    # Make sure your initialize_kernel returns (kernel, chat_completion) in that order.
    st.session_state["kernel"], st.session_state["chat_completion"], st.session_state["chat_history"] = initialize_kernel(
        st.session_state["kernel_config"]
    )

# Keep track of messages to display in bubbles
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

# Set page config with wide layout
st.set_page_config(page_title="MicroBot", layout="wide")

# Inject the custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

############################
# HEADER (centered)
############################

header_col1, header_col2, header_col3= st.columns([0.2, 0.6, 0.2])
with header_col1:
    pass
with header_col2:
    st.image("assets/assistant_logo.png", width=200)
    # Display the header with the image
    st.markdown("""
    <div id="centered-header" style="margin:0px;">
        <h2 style="margin:0px; color: #156082;">MicroBot</h2>
        <h3 style="margin:0px; color: #156082;">Microsoft’s Demo AI Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    st.image("assets/microsoft_logo.png", width=160)

############################
# SIDEBAR CONFIG
############################
with st.sidebar:
    st.markdown("### Website Tabs")
    view = st.radio(
        "Website Tabs",
        ["💬 Chat", "📂 Upload & Index"],
        key="selected_view",
        label_visibility="hidden"      # makes the header bold
    )
    st.markdown("---")

    st.markdown("### Model Parameters")
    max_tokens = st.slider("Max Output Tokens", min_value=10, max_value=1000, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-5-mini"])
    
    st.markdown("---")
    st.markdown("## Prompt Template Configuration")
    
    # 3 text boxes for context, filters, output format
    context_text = st.text_area("Context", value="")
    filters_text = st.text_area("Filters", value="")
    output_format_text = st.text_area("Output Format", value="")

    st.markdown("---")
    st.markdown("### Plugins")
    # (For now leave plugins aside – they will be empty in our kernel configuration.)
    weather_plugin = st.checkbox("Weather", value=False)
    time_plugin = st.checkbox("Time", value=False)
    math_plugin = st.checkbox("Math", value=False)
    internet_plugin = st.checkbox("Internet Search", value=False)

    if st.button("Apply Configuration", key="apply-config"):
        # Build the configuration with the expected key names
        config = {
            "selected_model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "context_text": context_text,
            "filters_text": filters_text,    
            "output_format_text": output_format_text,
            # Add plugins based on user selection
            "plugins": {
                "TimePlugin": time_plugin if time_plugin else None,
                "WeatherPlugin": weather_plugin if weather_plugin else None,
                "MathPlugin": math_plugin if math_plugin else None,
                "InternetSearchPlugin": internet_plugin if internet_plugin else None
            }
        }
        st.session_state["kernel_config"] = config
        # Re-initialize the kernel with the new configuration.
        st.session_state["kernel"], st.session_state["chat_completion"],  st.session_state["chat_history"] = initialize_kernel(config)
        st.session_state["messages"] = []  # Reset chat history
        st.success("Configuration applied. Chat history reset.")


if st.session_state.get("selected_view", "💬 Chat") == "💬 Chat":
    ############################
    # MAIN AREA - CHAT DISPLAY
    ############################
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            # Right-aligned (green bubble)
            st.markdown(
                f"""
                <div style="background-color:#2E8B57;; 
                            border-radius:10px; 
                            margin:5px 0; 
                            padding:8px 12px; 
                            display:inline-block;
                            max-width:80%;">
                    <strong>You:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Left-aligned (dark bubble)
            st.markdown(
                f"""
                <div style="background-color:#333333; 
                            border-radius:10px; 
                            margin:5px 0; 
                            padding:8px 12px; 
                            display:inline-block; 
                            max-width:80%;">
                    <strong>Assistant:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

    ############################
    # FIXED INPUT BAR AT THE BOTTOM
    ############################
    # The built-in chat_input is automatically pinned to the bottom of the page
    user_input = st.chat_input("Ask me anything…")

    if user_input:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

         # Retrieve kernel components
        kernel = st.session_state["kernel"]
        chat_completion = st.session_state["chat_completion"]
        chat_history = st.session_state["chat_history"]
        config = st.session_state["kernel_config"]

        # ----------- RAG only if memory exists -----------------
        if "memory" in st.session_state:
            memory = st.session_state["memory"]
            relevant_chunks = asyncio.run(
                retrieve_relevant_chunks(memory, user_input, top_k=3)
            )
            rag_context = "\n\n".join(f"• {c}" for c in relevant_chunks)
            final_user_input = (
                f"Refer to the following knowledge if helpful:\n{rag_context}\n\n"
                f"User question: {user_input}"
            )
        else:
            final_user_input = user_input
        # -------------------------------------------------------

        # Get assistant reply
        try:
            assistant_reply, chat_history = asyncio.run(
                get_reply(config, kernel, final_user_input, chat_history, chat_completion)
            )
        except Exception as e:
            assistant_reply = f"Kernel Error: {str(e)}"

        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply}) 

        st.rerun()


else:  # ------------- UPLOAD VIEW -------------
    ensure_memory()                       
    kernel  = st.session_state["kernel"]
    memory  = st.session_state["memory"]

    st.markdown("<h3>Add documents to your knowledge base</h3>", unsafe_allow_html=True)

    files = st.file_uploader(
        label="Drag‑drop or Browse",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if files:
        st.session_state["uploaded_files"] = files

    if st.button("Index selected files") and st.session_state["uploaded_files"]:
        ensure_memory()
        kernel  = st.session_state["kernel"]
        memory  = st.session_state["memory"]

        for f in st.session_state["uploaded_files"]:
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                reader = PdfReader(f)                         # f is a BytesIO
                text   = "".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")
            print(f"File {f.name} has {len(text)} characters")
            status = st.empty()                      # filename label
            bar    = st.progress(0.0)                # visual bar

            status.write(f"Embedding **{f.name}** …")

            # define a tiny callback that the worker will call
            def _update(frac):
                bar.progress(frac)

            asyncio.run(
                ingest_file(kernel, memory, f.name, text, on_progress=_update)
            )

            bar.empty()      # remove the bar
            status.success(f"Indexed {f.name} ✅")

        st.success("All files indexed! 🎉")
        st.session_state["uploaded_files"] = []


    # -----------------------------------------------------------------
    st.markdown("---")
    st.markdown("<h3>Add webpages or a GitHub repo</h3>", unsafe_allow_html=True)

    url_to_index = st.text_input("URL or GitHub repo")
    idx_col1, idx_col2 = st.columns(2)
    with idx_col1:
        index_web_btn = st.button("Index webpage")
    with idx_col2:
        index_repo_btn = st.button("Index GitHub repo")

    if index_web_btn and url_to_index:
        ensure_memory()
        with st.spinner("Fetching & embedding webpage…"):
            asyncio.run(ingest_webpage(kernel, memory, url_to_index))   # you already wrote this helper
        st.success("Webpage indexed!")

    if index_repo_btn and url_to_index:
        ensure_memory()
        with st.spinner("Crawling & embedding repo…"):
            asyncio.run(ingest_github_repo(kernel, memory, url_to_index))
        st.success("Repository indexed!")