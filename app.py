import streamlit as st
from query_data import RunRag
import os

with st.sidebar:
    "# Olá! 👋"
    "Esse é um chatbot desenvolvido para responder perguntas sobre fisiologia humana baseado nos livros texto mais utilizados na academia:"
    "Berne & Levy, Silverthorn, Guyton & Hall, Margarida Aires"
    "O chat funciona por meio de um processo chamado RAG Retrieved augmented generation, que é uma técnica de geração de texto baseada em modelos de linguagem onde o contexto é dado para o LLM utilizando busca por similaridade."
    "O modelo utilizado é o Gemini 1.5 Flash, que é um modelo de linguagem treinado pela Google, utilizando a API free tier, que tem uma limitação de chamadas e portanto pode falhar caso o número de requições seja elevado"
    "Esse projeto é um exemplo de como utilizar o RAG para responder perguntas de forma contextualizada e pode auxiliar no estudo da disciplina."
    "### Autor: Claucio Antonio Rank Filho"
    "Para saber mais:"
    st.markdown("""[![Open in GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://codespaces.new/streamlit/llm-examples?quickstart=1)
                <a href="https://codespaces.new/streamlit/llm-examples?quickstart=1">
                  <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="Open in LinkedIn" width="24" height="24">
                </a>
                """, unsafe_allow_html=True)

st.title("Fisiologia Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Faça sua pergunta!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write(RunRag().run(prompt, st.secrets["google_api_key"]))
        st.session_state.messages.append({"role": "assistant", "content": response})

    print(st.session_state.messages)
