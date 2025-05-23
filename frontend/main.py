import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Change if deploying elsewhere

st.set_page_config(page_title="Vanna NLP-to-SQL UI", layout="wide")
st.title("Vanna NLP-to-SQL System")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Connect to Database",
    "Train Model",
    "Query",
    "Agentic Query",
    "Add Question to RAG"
])

# Session state for session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Helper to show API errors
def show_error(resp):
    if resp is not None and resp.status_code != 200:
        st.error(f"API Error: {resp.status_code} - {resp.text}")

# 1. Connect to Database
if page == "Connect to Database":
    st.header("Connect to Database")
    with st.form("connect_form"):
        db_type = st.selectbox("Database Type", ["postgresql", "mysql", "mssql", "oracle", "sqlite", "snowflake", "bigquery", "redshift", "duckdb"])
        host = st.text_input("Host")
        port = st.text_input("Port")
        dbname = st.text_input("Database Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        connection_string = st.text_input("Connection String (optional)")
        project_id = st.text_input("Project ID (for BigQuery)")
        credentials_file = st.text_input("Credentials File (for BigQuery)")
        submitted = st.form_submit_button("Connect")
    if submitted:
        payload = {
            "db_type": db_type,
            "host": host or None,
            "port": int(port) if port else None,
            "dbname": dbname or None,
            "username": username or None,
            "password": password or None,
            "connection_string": connection_string or None,
            "project_id": project_id or None,
            "credentials_file": credentials_file or None
        }
        resp = requests.post(f"{API_URL}/connect", json=payload)
        if resp.status_code == 200:
            session_id = resp.json()["session_id"]
            st.session_state.session_id = session_id
            st.success(f"Connected! Session ID: {session_id}")
        else:
            show_error(resp)

# 2. Train Model
elif page == "Train Model":
    st.header("Train NLP-to-SQL Model")
    if not st.session_state.session_id:
        st.warning("Please connect to a database first.")
    else:
        with st.form("train_form"):
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            submitted = st.form_submit_button("Train Model")
        if submitted:
            payload = {
                "session_id": st.session_state.session_id,
                "openai_api_key": openai_api_key
            }
            resp = requests.post(f"{API_URL}/train", json=payload)
            if resp.status_code == 200:
                st.success("Training complete!")
            else:
                show_error(resp)

# 3. Query
elif page == "Query":
    st.header("Query the Database (NLP-to-SQL)")
    if not st.session_state.session_id:
        st.warning("Please connect to a database and train the model first.")
    else:
        with st.form("query_form"):
            question = st.text_area("Enter your question")
            return_sql_only = st.checkbox("Return SQL only", value=True)
            submitted = st.form_submit_button("Submit Query")
        if submitted:
            payload = {
                "session_id": st.session_state.session_id,
                "question": question,
                "return_sql_only": return_sql_only
            }
            resp = requests.post(f"{API_URL}/query", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                st.code(data.get("sql", ""), language="sql")
                if not return_sql_only and data.get("answer"):
                    st.write("Results:")
                    st.dataframe(data["answer"])
                if data.get("error"):
                    st.error(data["error"])
            else:
                show_error(resp)

# 4. Agentic Query
elif page == "Agentic Query":
    st.header("Agentic Query (Iterative Refinement)")
    if not st.session_state.session_id:
        st.warning("Please connect to a database and train the model first.")
    else:
        with st.form("agent_query_form"):
            question = st.text_area("Enter your question for agentic refinement")
            max_attempts = st.number_input("Max Attempts", min_value=1, max_value=10, value=5)
            submitted = st.form_submit_button("Submit Agentic Query")
        if submitted:
            payload = {
                "session_id": st.session_state.session_id,
                "question": question,
                "max_attempts": max_attempts
            }
            resp = requests.post(f"{API_URL}/agent-query", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                st.code(data.get("sql", ""), language="sql")
                if data.get("answer"):
                    st.write("Results:")
                    st.dataframe(data["answer"])
                if data.get("analysis"):
                    st.subheader("Actionable Analysis")
                    st.write(data["analysis"])
                if data.get("error"):
                    st.error(data["error"])
                st.info(f"Attempts: {data.get('attempts', 0)}")
            else:
                show_error(resp)

# 5. Add Question to RAG
elif page == "Add Question to RAG":
    st.header("Add Question to RAG Vector Store")
    if not st.session_state.session_id:
        st.warning("Please connect to a database and train the model first.")
    else:
        with st.form("add_question_form"):
            question = st.text_area("Enter a question to add to RAG")
            submitted = st.form_submit_button("Add Question")
        if submitted:
            payload = {
                "session_id": st.session_state.session_id,
                "question": question
            }
            resp = requests.post(f"{API_URL}/add-question", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                st.success(data.get("message", "Added!"))
                st.code(data.get("sql", ""), language="sql")
            else:
                show_error(resp)

st.sidebar.markdown("---")
st.sidebar.info("Developed for Vanna NLP-to-SQL API. See /docs for backend API documentation.") 