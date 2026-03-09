import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


# Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# Sidebar API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")


# URL input
generic_url = st.text_input("Enter YouTube or Website URL")


# Prompt template
prompt_template = """
Provide a summary of the following content in about 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# Button action
if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip():
        st.error("Please provide a Groq API Key")

    elif not generic_url.strip():
        st.error("Please enter a URL")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or Website)")

    else:
        try:
            with st.spinner("Loading content..."):

                # Initialize LLM
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=groq_api_key
                )

                # LCEL Chain
                chain = prompt | llm | StrOutputParser()

                # Load documents
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                docs = loader.load()

                # Combine text
                text = "\n".join(doc.page_content for doc in docs)
                text=text[0:1000]  # Limit text length

                # Run summarization
                summary = chain.invoke({"text": text})

                st.success("Summary Generated")
                st.write(summary)

        except Exception as e:
            st.error(f"Error: {e}")