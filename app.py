import streamlit as st

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

/* Position and style the 'Apply Configuration' button at bottom-right of sidebar */
[class="stElementContainer element-container st-key-apply-config st-emotion-cache-kj6hex eu6p4el1"]{
    position: fixed;
    bottom: 20px;
    color: #FFFFFF;
    border-radius: 5px;
    border: none;
    padding: 0.6rem 1rem;
}

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

[class="st-emotion-cache-130o8tb eu6p4el5"] {
    position:fixed;
    bottom:30px;
    width: 73vw;
}

/* Chat bubbles styling */
.user-message {
    text-align: right;
    background-color: #2E8B57; /* a green shade for user's bubble */
    border-radius: 10px;
    margin: 5px 0;
    padding: 8px 12px;
    display: inline-block;
    max-width: 80%;
}
.assistant-message {
    text-align: left;
    background-color: #333333;
    border-radius: 10px;
    margin: 5px 0;
    padding: 8px 12px;
    display: inline-block;
    max-width: 80%;
}
</style>
"""

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
    st.markdown("### Model Parameters")
    max_tokens = st.slider("Max Output Tokens", min_value=50, max_value=1000, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    
    st.markdown("---")
    st.markdown("### Plugins")
    location_plugin = st.checkbox("Location", value=True)
    weather_plugin = st.checkbox("Weather", value=True)
    time_plugin = st.checkbox("Time", value=False)

    if st.button("Apply Configuration", key="apply-config"):
        # Save configuration and reset chat
        st.session_state["kernel_config"] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "plugins": {
                "location": location_plugin,
                "weather": weather_plugin,
                "time": time_plugin
            }
        }
        st.session_state["messages"] = []
        st.success("Configuration applied. Chat history reset.")

############################
# MAIN AREA - CHAT DISPLAY
############################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

chat_container = st.container()
chat_container.markdown('<div id="chat-container">', unsafe_allow_html=True)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-message'><strong>You:</strong> {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='assistant-message'><strong>Assistant:</strong> {msg['content']}</div>",
            unsafe_allow_html=True
        )
chat_container.markdown('</div>', unsafe_allow_html=True)

############################
# FIXED INPUT BAR AT THE BOTTOM
############################
# Create an empty container to hold our custom HTML wrapper
input_section_placeholder = st.empty()

with st.container(height=100):
    # Create columns for the text input and send button.
    cols = st.columns([0.85, 0.15])
    with cols[0]:
        user_input = st.text_input("", key="real_chat_input", placeholder="Ask me something...", label_visibility="hidden")
    with cols[1]:
        # The Send button appears after the text input.
        send_button = st.button("Send", key="send_button")

# --- Process the Send button or non-empty input if needed ---
if (send_button or user_input.strip()) and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Simulate an assistant reply; replace this with your Semantic Kernel call.
    assistant_reply = f"Simulated reply for: {user_input}"
    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
    # Clear the text input by resetting its key via session_state.
    st.session_state["real_chat_input"] = ""
    st.experimental_rerun()