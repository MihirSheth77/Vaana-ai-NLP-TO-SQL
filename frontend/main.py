import streamlit as st
import requests
import time
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Enhanced Vanna NLP-to-SQL UI", layout="wide")
st.title("🚀 Enhanced Vanna NLP-to-SQL System")
st.markdown("*With Persistent Training Data & Smart Model Management*")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🔌 Connect to Database",
    "🧠 Train Model",
    "❓ Query",
    "🤖 Agentic Query",
    "📚 Add Question to RAG",
    "📊 Training Status",
    "🔄 Reload Model"
])

# Session state management
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "training_status" not in st.session_state:
    st.session_state.training_status = {}

# Helper functions
def show_error(resp):
    if resp is not None and resp.status_code != 200:
        st.error(f"API Error: {resp.status_code} - {resp.text}")
        return True
    return False

def get_training_status():
    if st.session_state.session_id:
        try:
            resp = requests.get(f"{API_URL}/training-status/{st.session_state.session_id}")
            if resp.status_code == 200:
                st.session_state.training_status = resp.json()
            return st.session_state.training_status
        except:
            return {}
    return {}

def show_training_status():
    status = get_training_status()
    if status:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Trained", "✅ Yes" if status.get("trained") else "❌ No")
        with col2:
            st.metric("Model Loaded", "✅ Loaded" if status.get("model_loaded") else "❌ Not Loaded")
        with col3:
            st.metric("Training Data Count", status.get("training_data_count", 0))

# 1. Connect to Database
if page == "🔌 Connect to Database":
    st.header("Connect to Database")
    
    # Show current connection status
    if st.session_state.session_id:
        st.success(f"✅ Connected! Session ID: {st.session_state.session_id}")
        show_training_status()
        st.info("You can train a new model or your existing trained model will be automatically loaded.")
    
    with st.form("connect_form"):
        db_type = st.selectbox("Database Type", [
            "postgresql", "mysql", "mssql", "oracle", "sqlite", 
            "snowflake", "bigquery", "redshift", "duckdb"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host")
            dbname = st.text_input("Database Name")
            username = st.text_input("Username")
        with col2:
            port = st.text_input("Port")
            password = st.text_input("Password", type="password")
            connection_string = st.text_input("Connection String (optional)")
        
        # Special fields for specific databases
        if db_type == "bigquery":
            project_id = st.text_input("Project ID (for BigQuery)")
            credentials_file = st.text_input("Credentials File (for BigQuery)")
        else:
            project_id = ""
            credentials_file = ""
        
        submitted = st.form_submit_button("🔌 Connect")
    
    if submitted:
        # Validate port input
        port_value = None
        if port:
            if port.isdigit():
                port_value = int(port)
            else:
                st.error(f"Invalid port value: '{port}'. Port must be a number.")
                st.stop()
        
        payload = {
            "db_type": db_type,
            "host": host or None,
            "port": port_value,
            "dbname": dbname or None,
            "username": username or None,
            "password": password or None,
            "connection_string": connection_string or None,
            "project_id": project_id or None,
            "credentials_file": credentials_file or None
        }
        
        with st.spinner("Connecting to database..."):
            resp = requests.post(f"{API_URL}/connect", json=payload)
            
        if resp.status_code == 200:
            data = resp.json()
            session_id = data["session_id"]
            st.session_state.session_id = session_id
            st.success(f"✅ {data['message']}")
            get_training_status()  # Refresh status
            st.rerun()
        else:
            show_error(resp)

# 2. Train Model
elif page == "🧠 Train Model":
    st.header("Train NLP-to-SQL Model")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please connect to a database first.")
    else:
        # Show current status
        status = get_training_status()
        show_training_status()
        
        if status.get("trained"):
            st.info("🎉 Model is already trained! You can:")
            st.markdown("- Use it for queries directly")
            st.markdown("- Retrain to add more data")
            st.markdown("- Add new questions to improve it")
        
        with st.form("train_form"):
            openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                         help="Required for training and using the model")
            force_retrain = st.checkbox("Force Retrain", 
                                      help="Retrain even if model already exists")
            submitted = st.form_submit_button("🧠 Train Model")
        
        if submitted:
            if not openai_api_key:
                st.error("OpenAI API Key is required!")
            else:
                payload = {
                    "session_id": st.session_state.session_id,
                    "openai_api_key": openai_api_key
                }
                
                with st.spinner("Training model... This may take a few minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Show progress updates
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("Extracting database schema...")
                        elif i < 60:
                            status_text.text("Generating example queries...")
                        elif i < 90:
                            status_text.text("Training AI model...")
                        else:
                            status_text.text("Finalizing...")
                        time.sleep(0.1)
                    
                    resp = requests.post(f"{API_URL}/train", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data['message']}")
                    get_training_status()  # Refresh status
                else:
                    show_error(resp)

# 3. Query
elif page == "❓ Query":
    st.header("Query the Database (NLP-to-SQL)")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please connect to a database first.")
    else:
        status = get_training_status()
        show_training_status()
        
        if not status.get("trained"):
            st.warning("⚠️ Please train the model first.")
        elif not status.get("model_loaded"):
            st.warning("⚠️ Model not loaded. Please use 'Reload Model' page to load your trained model.")
        else:
            st.success("🎉 Ready to answer your questions!")
            
            # Quick examples
            st.markdown("### 💡 Example Questions:")
            example_questions = [
                "Show me the top 10 customers by revenue",
                "What are the most popular products this month?",
                "How many orders were placed yesterday?",
                "Show me sales by region",
                "What's the average order value?"
            ]
            
            selected_example = st.selectbox("Quick Examples", [""] + example_questions)
            
            with st.form("query_form"):
                question = st.text_area("Enter your question", 
                                       value=selected_example,
                                       height=100,
                                       help="Ask questions in natural language about your data")
                return_sql_only = st.checkbox("Return SQL only", value=False)
                submitted = st.form_submit_button("🔍 Submit Query")
            
            if submitted and question:
                payload = {
                    "session_id": st.session_state.session_id,
                    "question": question,
                    "return_sql_only": return_sql_only
                }
                
                with st.spinner("Generating SQL and fetching results..."):
                    resp = requests.post(f"{API_URL}/query", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if data.get("error"):
                        st.error(f"Error: {data['error']}")
                    else:
                        # Show SQL
                        st.subheader("Generated SQL:")
                        st.code(data.get("sql", ""), language="sql")
                        
                        # Show results
                        if not return_sql_only and data.get("answer"):
                            st.subheader("Results:")
                            st.dataframe(data["answer"], use_container_width=True)
                            st.info(f"Found {len(data['answer'])} rows")
                else:
                    show_error(resp)

# 4. Agentic Query
elif page == "🤖 Agentic Query":
    st.header("Agentic Query (Iterative Refinement)")
    st.markdown("*Advanced AI that iteratively refines queries and provides analysis*")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please connect to a database first.")
    else:
        status = get_training_status()
        show_training_status()
        
        if not status.get("trained") or not status.get("model_loaded"):
            st.warning("⚠️ Please ensure your model is trained and loaded.")
        else:
            with st.form("agent_query_form"):
                question = st.text_area("Enter your question for agentic refinement",
                                      height=100,
                                      help="The AI will iteratively refine the query if errors occur")
                max_attempts = st.number_input("Max Attempts", min_value=1, max_value=10, value=5)
                submitted = st.form_submit_button("🤖 Submit Agentic Query")
            
            if submitted and question:
                payload = {
                    "session_id": st.session_state.session_id,
                    "question": question,
                    "max_attempts": max_attempts
                }
                
                with st.spinner("AI is working on your query..."):
                    resp = requests.post(f"{API_URL}/agent-query", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if data.get("error"):
                        st.error(f"Error: {data['error']}")
                    else:
                        # Show results
                        col1, col2 = st.columns([2, 1])
                        with col2:
                            st.metric("Attempts Used", data.get("attempts", 0))
                        
                        st.subheader("Generated SQL:")
                        st.code(data.get("sql", ""), language="sql")
                        
                        if data.get("answer"):
                            st.subheader("Results:")
                            st.dataframe(data["answer"], use_container_width=True)
                        
                        if data.get("analysis"):
                            st.subheader("🎯 Actionable Analysis:")
                            st.write(data["analysis"])
                else:
                    show_error(resp)

# 5. Add Question to RAG
elif page == "📚 Add Question to RAG":
    st.header("Add Question to RAG Vector Store")
    st.markdown("*Improve future responses by adding new question-answer pairs*")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please connect to a database first.")
    else:
        status = get_training_status()
        show_training_status()
        
        if not status.get("trained") or not status.get("model_loaded"):
            st.warning("⚠️ Please ensure your model is trained and loaded.")
        else:
            st.info("💡 This will generate SQL for your question and add both to the training data for better future responses.")
            
            with st.form("add_question_form"):
                question = st.text_area("Enter a question to add to RAG",
                                      height=100,
                                      help="This question will be added to improve future similar queries")
                submitted = st.form_submit_button("📚 Add Question")
            
            if submitted and question:
                payload = {
                    "session_id": st.session_state.session_id,
                    "question": question
                }
                
                with st.spinner("Generating SQL and adding to RAG..."):
                    resp = requests.post(f"{API_URL}/add-question", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data.get('message', 'Added!')}")
                    
                    if data.get("sql"):
                        st.subheader("Generated SQL:")
                        st.code(data.get("sql", ""), language="sql")
                        st.info("This question-SQL pair has been added to your model's training data.")
                else:
                    show_error(resp)

# 6. Training Status
elif page == "📊 Training Status":
    st.header("Training Status & Model Information")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please connect to a database first.")
    else:
        # Refresh button
        if st.button("🔄 Refresh Status"):
            get_training_status()
        
        status = get_training_status()
        
        if status:
            # Main status display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Trained", "✅ Yes" if status.get("trained") else "❌ No")
            with col2:
                st.metric("Model Loaded", "✅ Loaded" if status.get("model_loaded") else "❌ Not Loaded")
            with col3:
                st.metric("Training Data Count", status.get("training_data_count", 0))
            with col4:
                st.metric("Session Active", "✅ Yes" if st.session_state.session_id else "❌ No")
            
            # Detailed information
            st.subheader("Model Details")
            info_data = {
                "Session ID": st.session_state.session_id,
                "Model ID": status.get("model_id", "N/A"),
                "Model Path": status.get("model_path", "N/A"),
                "Training Status": "Trained" if status.get("trained") else "Not Trained",
                "Load Status": "Loaded" if status.get("model_loaded") else "Not Loaded"
            }
            
            for key, value in info_data.items():
                st.text(f"{key}: {value}")
            
            # Show all available models
            st.subheader("All Available Models")
            try:
                resp = requests.get(f"{API_URL}/list-models")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    if models:
                        st.write(f"Found {len(models)} trained models:")
                        for model in models:
                            st.text(f"Model ID: {model['model_id']}")
                            st.text(f"Path: {model['path']}")
                            st.text("---")
                    else:
                        st.info("No trained models found.")
                else:
                    st.error("Failed to fetch model list.")
            except Exception as e:
                st.error(f"Error fetching models: {e}")

            # Reload Model button and logic
            if status.get("trained") and not status.get("model_loaded"):
                st.warning("Please ensure your model is trained and loaded.")
                openai_api_key = st.text_input("OpenAI API Key to reload model", type="password", key=f"reload_{st.session_state.session_id}")
                if st.button("Reload Model", key=f"reload_btn_{st.session_state.session_id}"):
                    if not openai_api_key:
                        st.error("Please enter your OpenAI API key.")
                    else:
                        with st.spinner("Reloading model from disk..."):
                            reload_resp = requests.post(f"{API_URL}/reload-model", json={"session_id": st.session_state.session_id, "openai_api_key": openai_api_key})
                        if reload_resp.status_code == 200:
                            st.success("Model reloaded successfully! Please refresh status.")
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to reload model: {reload_resp.text}")
        else:
            st.info("No status information available. Please connect to a database first.")

# 7. Reload Model
elif page == "🔄 Reload Model":
    st.header("Reload Trained Model")
    st.markdown("*Load a previously trained model from persistent storage*")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please connect to a database first.")
    else:
        status = get_training_status()
        show_training_status()
        
        if not status.get("trained"):
            st.warning("⚠️ No trained model found for this database connection. Please train a model first.")
        elif status.get("model_loaded"):
            st.success("✅ Model is already loaded and ready to use!")
        else:
            st.info("📁 Found a trained model that needs to be loaded.")
            
            with st.form("reload_form"):
                st.markdown("**Why do I need to reload?**")
                st.markdown("- The backend was restarted")
                st.markdown("- You switched to a different session")
                st.markdown("- The model was trained in a previous session")
                
                openai_api_key = st.text_input("OpenAI API Key", type="password",
                                             help="Required to reload and use the model")
                submitted = st.form_submit_button("🔄 Reload Model")
            
            if submitted:
                if not openai_api_key:
                    st.error("OpenAI API Key is required to reload the model!")
                else:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "openai_api_key": openai_api_key
                    }
                    
                    with st.spinner("Reloading trained model..."):
                        resp = requests.post(f"{API_URL}/reload-model", json=payload)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"✅ {data['message']}")
                        get_training_status()  # Refresh status
                        st.balloons()
                    else:
                        show_error(resp)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Enhanced Features")
st.sidebar.markdown("✅ Persistent model storage")
st.sidebar.markdown("✅ Automatic model reloading")
st.sidebar.markdown("✅ Smart session management")
st.sidebar.markdown("✅ Training status tracking")
st.sidebar.markdown("✅ ChromaDB vector storage")

st.sidebar.markdown("---")
st.sidebar.info("Developed for Enhanced Vanna NLP-to-SQL API. See /docs for backend API documentation.")

# Auto-refresh status in the background (optional)
if st.session_state.session_id and 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Refresh every 30 seconds if on status page
if page == "📊 Training Status" and st.session_state.session_id:
    current_time = time.time()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = current_time
    elif current_time - st.session_state.last_refresh > 30:
        get_training_status()
        st.session_state.last_refresh = current_time