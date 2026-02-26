import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

st.title("Conversational RAG With PDF Uploads")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        documents = []

        for uploaded_file in uploaded_files:
            temp_path = f"./{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)

        # Vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a helpful assistant. "
                 "Answer the question using only the provided context. "
                 "If the answer is not in the context, say you don't know.\n\n"
                 "Context:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}")
            ]
        )

    
        rag_chain = (
            RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["question"])
        )
        
        | prompt
        | llm
        | StrOutputParser()
    )

        # Session-based chat history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        user_input = st.text_input("Ask a question:")

        if user_input:
            response = conversational_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("Assistant:", response)