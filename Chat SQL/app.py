import streamlit as st
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine

from langchain_groq import ChatGroq

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from langchain.agents import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = [
    "Use SQLite3 Database - student.db",
    "Connect to your MySQL Database"
]

selected_opt = st.sidebar.radio(
    label="Choose the DB which you want to chat",
    options=radio_opt
)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not api_key:
    st.info("Please add the Groq API key")

# LLM
llm = ChatGroq( 
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    streaming=True
)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):

    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()

        creator = lambda: sqlite3.connect(
            f"file:{dbfilepath}?mode=ro", uri=True
        )

        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    elif db_uri == MYSQL:

        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):

        streamlit_callback = StreamlitCallbackHandler(st.container())

        response = agent.invoke(
            {"input": user_query},
            {"callbacks": [streamlit_callback]}
        )

        output = response["output"]

        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.write(output)