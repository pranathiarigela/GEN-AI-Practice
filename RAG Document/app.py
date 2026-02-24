import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------
# LLM Setup (Groq)
# -------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# -------------------------
# Prompt Template
# -------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the provided context.
    If the answer is not found in the context, say you don't know.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# -------------------------
# Create Vector Embeddings
# -------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(docs[:50])

        st.session_state.vectors = FAISS.from_documents(
            split_docs,
            st.session_state.embeddings
        )


# -------------------------
# Streamlit UI
# -------------------------
st.title("RAG Document Q&A With Groq And Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Vector Database is ready")

# -------------------------
# RAG Execution (LangChain 1.x)
# -------------------------
if user_prompt and "vectors" in st.session_state:

    retriever = st.session_state.vectors.as_retriever()

    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    response = rag_chain.invoke(user_prompt)
    end = time.process_time()

    st.write(response)
    st.caption(f"Response time: {end - start:.2f} seconds")

    # Show similar documents
    with st.expander("Document Similarity Search"):
        docs = retriever.invoke(user_prompt)        
        for doc in docs:
            st.write(doc.page_content)
            st.write("----------------------------")