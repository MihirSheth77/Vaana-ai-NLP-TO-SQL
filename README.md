# 🚀 Enhanced Vanna NLP-to-SQL System

A robust, production-ready system for natural language to SQL translation, featuring:
- **BEAST MODE**: Generates hundreds of high-quality training examples for maximum accuracy
- **Persistent Training Data**: Models and connection info are stored locally (but excluded from git)
- **Smart Model Management**: Reload trained models after restart, force retrain, and session restore
- **Modern Frontend**: Streamlit UI for multi-session management, query, agentic refinement, and context doc upload

---

## Features
- 🔥 **BEAST MODE**: Massive, diverse training data generation
- 💾 **Persistent Storage**: Models and connections saved in `vanna_storage/` (excluded from git)
- 🧠 **Smart Model Loading**: Reload models after backend restart without retraining
- 🗂️ **Multi-Session Support**: Manage multiple database connections and models
- 📝 **Contextual Document Upload**: Enhance model with business docs, transcripts, etc.
- 🤖 **Agentic Query Refinement**: Iterative, error-aware SQL generation
- 📊 **Query History**: Track all queries and results per session
- 🛡️ **Secure**: API keys and sensitive data never committed to git

---

## Quickstart

1. **Clone the repo and install dependencies**
2. **Run the backend**
   ```sh
   uvicorn api_vanna:app --reload
   ```
3. **Run the frontend**
   ```sh
   streamlit run frontend/main.py
   ```
4. **Connect to your database** via the UI
5. **Train the model** (BEAST MODE!)
6. **Query, refine, and analyze!**

---

## Directory Structure & .gitignore
- All persistent models and connection info are stored in:
  - `vanna_storage/models/`
  - `vanna_storage/connections/`
- These folders are **excluded from git** (see `.gitignore`) to protect sensitive data.

---

## Usage

### Backend (FastAPI)
- Endpoints for connect, train, query, agentic query, add-question, upload-context, reload-model, and session management.
- See `/docs` for full OpenAPI documentation when running.

### Frontend (Streamlit)
- Multi-session sidebar: create, select, rename, delete sessions
- Tabs for: Train Model, Query, Agentic Query, Add Question, Upload Context Docs, Query History
- Model status indicators: trained, loaded, training data count
- Reload Model button for post-restart recovery

---

## Model Training & Loading
- **Train**: Use BEAST MODE to generate and ingest hundreds of examples
- **Reload**: After backend restart, use the Reload Model button (provide OpenAI API key)
- **Force Retrain**: Optionally retrain to add more data or refresh the model

---

## Troubleshooting
- **Model Not Loaded after Restart?** Use the Reload Model button and provide your OpenAI API key.
- **Training Data Count is Low?** Force retrain or check your database schema.
- **Sensitive Data in Git?** Ensure `vanna_storage/connections/` and `vanna_storage/models/` are in `.gitignore` (they are by default).
- **Session Lost after Restart?** Use the restore session feature or reconnect via the UI.

---

## Contributing
PRs welcome! Please ensure you do not commit any sensitive data or API keys.

---

## License
MIT 