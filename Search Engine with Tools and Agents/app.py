import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------
# Tools
# ---------------------------

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="search")

tools = [search, arxiv, wiki]

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🔎 LangChain 1.2 - Chat with Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ---------------------------
# Chat History
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search Wikipedia, Arxiv and the web. Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------
# User Input
# ---------------------------

if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # ---------------------------
    # LLM
    # ---------------------------

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",   # updated model
        streaming=True
    )

    # ---------------------------
    # Agent (LangChain 1.2)
    # ---------------------------

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant that can search the web, Wikipedia and Arxiv."
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        response = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            config={"callbacks": [st_cb]}
        )

        answer = response["messages"][-1].content

        st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )