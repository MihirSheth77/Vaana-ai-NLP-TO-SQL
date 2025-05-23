from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import threading
import uuid
import os

# Import Vanna logic from train_vanna_rds.py
from train_vanna_rds import VannaTrainer, get_connection_string, extract_schema, extract_example_queries
import sqlalchemy
from sqlalchemy import text

app = FastAPI(title="Vanna NLP-to-SQL API", description="API for NLP-to-SQL with RAG and training endpoints.", version="1.0.0")

# In-memory user session store (for demo; use Redis/DB for production)
user_sessions = {}
session_lock = threading.Lock()

# ------------------- Pydantic Models ------------------- #
class ConnectRequest(BaseModel):
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    dbname: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    project_id: Optional[str] = None
    credentials_file: Optional[str] = None

class ConnectResponse(BaseModel):
    session_id: str
    message: str

class TrainRequest(BaseModel):
    session_id: str
    openai_api_key: str

class TrainResponse(BaseModel):
    message: str

class QueryRequest(BaseModel):
    session_id: str
    question: str
    return_sql_only: Optional[bool] = True

class QueryResponse(BaseModel):
    sql: Optional[str] = None
    answer: Optional[Any] = None
    error: Optional[str] = None

class AddQuestionRequest(BaseModel):
    session_id: str
    question: str

class AddQuestionResponse(BaseModel):
    message: str
    sql: Optional[str] = None

class AgentQueryRequest(BaseModel):
    session_id: str
    question: str
    max_attempts: Optional[int] = 5

class AgentQueryResponse(BaseModel):
    sql: Optional[str] = None
    answer: Optional[Any] = None
    analysis: Optional[str] = None
    attempts: int
    error: Optional[str] = None

# ------------------- Dependency ------------------- #
def get_vanna_session(session_id: str) -> Dict[str, Any]:
    with session_lock:
        session = user_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found. Please connect first.")
        return session

# ------------------- Endpoints ------------------- #
@app.post("/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest):
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    # Build args object for get_connection_string
    class Args: pass
    args = Args()
    for field, value in req.dict().items():
        setattr(args, field, value)
    # Get connection string
    try:
        conn_string = get_connection_string(args)
        engine = sqlalchemy.create_engine(conn_string)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection failed: {e}")
    # Store session
    with session_lock:
        user_sessions[session_id] = {
            "args": args,
            "conn_string": conn_string,
            "engine": engine,
            "vn": None,  # Will be set after training
            "trained": False
        }
    return ConnectResponse(session_id=session_id, message="Connection established. Use this session_id for further requests.")

BATCH_SIZE = 20  # Number of DDLs/examples per batch to avoid context/token limit issues

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    session = get_vanna_session(req.session_id)
    engine = session["engine"]
    args = session["args"]
    # Extract schema and example queries
    try:
        ddl_statements = extract_schema(engine, args.db_type)
        example_queries = extract_example_queries(engine, args.db_type)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Schema extraction failed: {e}")
    # Train Vanna in batches to avoid context/token limit errors
    try:
        vn = VannaTrainer(config={
            'api_key': req.openai_api_key,
            'model': 'gpt-4o'  # or 'gpt-4o-high' if supported
        })
        # Batch training for DDLs
        for i in range(0, len(ddl_statements), BATCH_SIZE):
            batch = ddl_statements[i:i+BATCH_SIZE]
            for ddl in batch:
                vn.train(ddl=ddl)
        # Batch training for example queries
        for i in range(0, len(example_queries), BATCH_SIZE):
            batch = example_queries[i:i+BATCH_SIZE]
            for query in batch:
                vn.train(sql=query)
        session["vn"] = vn
        session["trained"] = True
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
    return TrainResponse(message="Training complete.")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    session = get_vanna_session(req.session_id)
    if not session.get("trained") or not session.get("vn"):
        return QueryResponse(error="Model not trained. Please call /train first.")
    vn = session["vn"]
    try:
        sql = vn.ask(req.question)
        if req.return_sql_only:
            return QueryResponse(sql=sql)
        # Optionally, run the SQL and return results
        engine = session["engine"]
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(sql))
            rows = result.fetchall()
            columns = result.keys()
            answer = [dict(zip(columns, row)) for row in rows]
        return QueryResponse(sql=sql, answer=answer)
    except Exception as e:
        return QueryResponse(error=str(e))

@app.post("/add-question", response_model=AddQuestionResponse)
def add_question(req: AddQuestionRequest):
    session = get_vanna_session(req.session_id)
    if not session.get("trained") or not session.get("vn"):
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    vn = session["vn"]
    try:
        sql = vn.ask(req.question)
        vn.train(question=req.question, sql=sql)
        return AddQuestionResponse(message="Question added to RAG vector store.", sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question: {e}")

@app.post("/agent-query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest):
    """
    Agentic, iterative query refinement with error analysis.
    Iteratively generates and refines SQL until a valid result is returned or max_attempts is reached.
    Returns the final SQL, result, analysis, and number of attempts.
    """
    session = get_vanna_session(req.session_id)
    if not session.get("trained") or not session.get("vn"):
        return AgentQueryResponse(error="Model not trained. Please call /train first.", attempts=0)
    vn = session["vn"]
    engine = session["engine"]
    context = ""
    sql = None
    answer = None
    analysis = None
    last_error = None
    for attempt in range(1, req.max_attempts + 1):
        # Add error context to the prompt if any
        prompt = req.question if not context else f"{req.question}\n\nError encountered: {context}"
        try:
            sql = vn.ask(prompt)
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql))
                rows = result.fetchall()
                columns = result.keys()
                answer = [dict(zip(columns, row)) for row in rows]
            # Generate analysis/actionable output for developers
            analysis_prompt = f"Given the following data (columns: {columns}, rows: {answer[:5]}), draft actionable steps for a developer to implement based on the user's question: '{req.question}'."
            analysis = vn.ask(analysis_prompt)
            return AgentQueryResponse(sql=sql, answer=answer, analysis=analysis, attempts=attempt)
        except Exception as e:
            context = str(e)
            last_error = context
    return AgentQueryResponse(error=f"Failed after {req.max_attempts} attempts. Last error: {last_error}", attempts=req.max_attempts)

# ------------------- Root Endpoint ------------------- #
@app.get("/")
def root():
    return {"message": "Welcome to the Vanna NLP-to-SQL API. See /docs for OpenAPI documentation."} 