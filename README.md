# 🤖 Microbot

Microbot is an AI chatbot demo built with [Semantic Kernel (SK)](https://aka.ms/semantic-kernel), [Streamlit](https://streamlit.io/), and [OpenAI](https://openai.com/) to showcase the main concepts and functioning of basic Gen AI Concepts.

## 🚀 Features

- 🔍 **Index GitHub repositories** using embeddings
- 🧠 **Ask questions** about the indexed content
- 🗂️ Uses **semantic memory** with chunking and vector search
- 🛠️ Built on **OpenAI + Azure OpenAI embeddings**
- 🌐 Web-based UI powered by Streamlit

## 📦 Requirements

- Python 3.10+
- `semantic-kernel==1.28.1`
- Streamlit
- OpenAI SDK
- GitPython

## 📁 Project Structure
<pre> Microbot/ 
├── assets/ 
│ └── ConnectEmployeeGuide.pdf 
├── src/ │ 
├── .env.template 
│ ├── indexing.py 
│ ├── kernel_utils.py 
│ └── myskills.py 
├── .gitignore 
├── requirements.txt 
├── README.md 
├── app.py 
</pre>


## 🧪 Run the App

1. Clone the repo:
```bash
    git clone https://github.com/Android998/MicrobotDemo.git
    cd MicrobotDemo
```

2. Create a virtual environment and install dependencies:
```bash
    python -m venv venv
    venv\Scripts\activate   # On Windows
    pip install -r requirements.txt
```

3. Run the app:
```bash
    streamlit run app.py
```  


## 🔐 Authentication
- You’ll need an OpenAI API key or Azure OpenAI key with access to an embedding model (e.g., text-embedding-ada-002).
- Store keys in environment variables in an .env file inside src folder (you can find there a template).