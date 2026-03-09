import streamlit as st
from langchain_groq import ChatGroq

from langchain_core.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# Streamlit setup
st.set_page_config(
    page_title="Math Problem Solver & Search Assistant",
    page_icon="🧮"
)

st.title("🧮 Math Problem Solver using Llama 3")

groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()


# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)


# Wikipedia Tool
wiki = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Useful for searching factual information from Wikipedia."
)


# Math + Reasoning Tool
def math_reasoning_tool(question: str):
    prompt = f"""
    Solve the following question step by step.

    Question: {question}

    Give a clear explanation and final answer.
    """
    return llm.invoke(prompt).content


reasoning_tool = Tool(
    name="Math Reasoning",
    func=math_reasoning_tool,
    description="Use this for solving math problems and logical reasoning questions."
)


# Agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! I can help with math problems and general knowledge questions."}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# User input
question = st.text_area(
    "Enter your question:",
    "If I have 5 bananas and eat 2, how many remain?"
)


if st.button("Find my answer"):

    if question:

        with st.spinner("Generating response..."):

            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container())

            response = assistant_agent.run(
                question,
                callbacks=[st_cb]
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter the question.")