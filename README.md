# Vanna.AI NLP-to-SQL System

This project provides an end-to-end solution for training and deploying an NLP-to-SQL system using Vanna.AI, with support for large, real-world databases and modern LLMs (e.g., GPT-4o). It features a FastAPI backend, a Streamlit frontend, and robust batching for scalable training.

## Features

- **Universal Database Support**: Connect to almost any SQL database (PostgreSQL, MySQL, SQL Server, Oracle, Snowflake, BigQuery, etc.)
- **Automatic Schema Extraction**: Extracts table definitions, columns, and relationships
- **Example Query Generation**: Creates example queries for better training
- **Batch Training**: Handles large schemas by batching DDLs and examples to avoid LLM context/token limits
- **RAG-Ready Architecture**: Retrieval-Augmented Generation for efficient query-time context
- **FastAPI Backend**: REST API for connect, train, query, agentic query, and RAG question addition
- **Streamlit Frontend**: User-friendly UI for all major workflows
- **Interactive Query Interface**: Ask questions and see the SQL + results

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the Backend (FastAPI)
```bash
uvicorn api_vanna:app --reload
```
- The API will be available at http://localhost:8000
- See interactive docs at http://localhost:8000/docs

### 2. Start the Frontend (Streamlit)
```bash
streamlit run frontend/main.py
```
- The UI will open in your browser.

## API Endpoints
- `POST /connect` — Connect to a database
- `POST /train` — Train the NLP-to-SQL model (with batching for large schemas)
- `POST /query` — Run a natural language query (returns SQL and/or results)
- `POST /agent-query` — Agentic, iterative query refinement with error analysis
- `POST /add-question` — Add a question to the RAG vector store

## Supported Database Types
- `postgresql`, `mysql`, `mssql`, `oracle`, `sqlite`, `snowflake`, `bigquery`, `redshift`, `duckdb`

## OpenAI API Key & Model
- You must provide a valid OpenAI API key for training and querying.
- The default model is `gpt-4o` (or `gpt-4o-high` if supported by your provider).
- Make sure your API key has sufficient quota ([check here](https://platform.openai.com/usage)).

## Example Workflow
1. **Connect to your database** via the frontend or `/connect` endpoint.
2. **Train the model** with your OpenAI API key (handles large schemas via batching).
3. **Ask questions** in natural language and get SQL or results.
4. **Use agentic query** for iterative, error-aware SQL refinement.
5. **Add custom questions** to the RAG vector store for improved retrieval.

## Troubleshooting

- **Large schema errors:** The system now batches DDLs and examples to avoid context/token limits. If you still hit errors, try reducing batch size in `api_vanna.py` (`BATCH_SIZE`).
- **Model errors:** Ensure your OpenAI API key has access to the specified model (`gpt-4o`, `gpt-4o-high`, etc.).
- **General FastAPI errors:** See [FastAPI error handling docs](https://fastapi.tiangolo.com/tutorial/handling-errors/).

## Notes
- Your database contents are never sent to the LLM—only schema and example queries are used for training.
- Vector database storage is handled through ChromaDB by default.
- The system is designed for extensibility: you can add new endpoints, models, or RAG strategies as needed.

---

For more details, see the code and comments in `api_vanna.py` and `frontend/main.py`. 