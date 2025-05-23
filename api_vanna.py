from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import threading
import uuid
import os
import json
import hashlib
from pathlib import Path
import getpass
import sqlalchemy
from sqlalchemy import inspect, text

# Import the BEAST MODE trainer
from beast_mode_trainer import BeastModeVannaTrainer, generate_model_id

app = FastAPI(title="BEAST MODE Vanna NLP-to-SQL API", description="🔥 BEAST MODE API for NLP-to-SQL with 500+ training examples", version="2.0.0")

# Create persistent storage directories
STORAGE_DIR = Path("vanna_storage")
MODELS_DIR = STORAGE_DIR / "models"
CONNECTIONS_DIR = STORAGE_DIR / "connections"

# Create directories if they don't exist
STORAGE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CONNECTIONS_DIR.mkdir(exist_ok=True)

# ------------------- Utility Functions ------------------- #

def get_connection_string(args) -> str:
    """Build a connection string from provided arguments"""
    # Handle missing password attribute gracefully
    password = getattr(args, 'password', None) or getpass.getpass("Database password: ")
    
    if hasattr(args, 'connection_string') and args.connection_string:
        return args.connection_string
    
    # Build connection string based on database type
    if args.db_type == 'postgresql':
        return f"postgresql://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}"
    elif args.db_type == 'mysql':
        return f"mysql+pymysql://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}"
    elif args.db_type == 'mssql':
        return f"mssql+pyodbc://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}?driver=ODBC+Driver+17+for+SQL+Server"
    elif args.db_type == 'oracle':
        return f"oracle+cx_oracle://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}"
    elif args.db_type == 'sqlite':
        return f"sqlite:///{args.dbname}"
    elif args.db_type == 'snowflake':
        account = args.host.split('.')[0] if '.' in args.host else args.host
        return f"snowflake://{args.username}:{password}@{account}/{args.dbname}"
    elif args.db_type == 'bigquery':
        if hasattr(args, 'credentials_file') and args.credentials_file:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials_file
        return f"bigquery://{args.project_id}/{args.dbname}"
    elif args.db_type == 'redshift':
        return f"redshift+psycopg2://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}"
    elif args.db_type == 'duckdb':
        return f"duckdb:///{args.dbname}"
    else:
        raise ValueError(f"Unsupported database type: {args.db_type}")

def extract_schema(engine, db_type: str) -> List[str]:
    """Extract schema based on database type"""
    print(f"Extracting schema for {db_type}...")
    if db_type == 'postgresql':
        return extract_ddl_postgres(engine)
    elif db_type == 'mysql':
        return extract_ddl_mysql(engine)
    elif db_type == 'mssql':
        return extract_ddl_mssql(engine)
    else:
        return extract_ddl_generic(engine)

def extract_ddl_postgres(engine) -> List[str]:
    """Extract DDL statements from PostgreSQL"""
    inspector = inspect(engine)
    ddl_statements = []
    
    schemas = inspector.get_schema_names()
    for schema in schemas:
        if schema in ('pg_catalog', 'information_schema'):
            continue
            
        tables = inspector.get_table_names(schema=schema)
        for table in tables:
            with engine.connect() as conn:
                query = text(f"""
                SELECT 
                    'CREATE TABLE ' || table_schema || '.' || table_name || ' (' ||
                    string_agg(column_name || ' ' || data_type || 
                        CASE WHEN character_maximum_length IS NOT NULL 
                             THEN '(' || character_maximum_length || ')' 
                             ELSE '' 
                        END ||
                        CASE WHEN is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END,
                        ', ') ||
                    ');' as create_statement
                FROM 
                    information_schema.columns
                WHERE 
                    table_schema = '{schema}' AND 
                    table_name = '{table}'
                GROUP BY 
                    table_schema, table_name;
                """)
                result = conn.execute(query)
                for row in result:
                    ddl_statements.append(row[0])
    
    return ddl_statements

def extract_ddl_mysql(engine) -> List[str]:
    """Extract DDL statements from MySQL"""
    inspector = inspect(engine)
    ddl_statements = []
    
    tables = inspector.get_table_names()
    for table in tables:
        with engine.connect() as conn:
            query = text(f"SHOW CREATE TABLE `{table}`")
            result = conn.execute(query)
            for row in result:
                ddl_statements.append(row[1])
    
    return ddl_statements

def extract_ddl_mssql(engine) -> List[str]:
    """Extract DDL statements from MS SQL Server"""
    inspector = inspect(engine)
    ddl_statements = []
    
    tables = inspector.get_table_names()
    for table in tables:
        with engine.connect() as conn:
            query = text(f"""
            SELECT 
                'CREATE TABLE [' + s.name + '].[' + t.name + '] (' + 
                STRING_AGG(
                    CAST('[' + c.name + '] ' + 
                    tp.name + 
                    CASE 
                        WHEN tp.name IN ('varchar', 'nvarchar', 'char', 'nchar') 
                        THEN '(' + CAST(c.max_length AS VARCHAR) + ')' 
                        ELSE '' 
                    END + 
                    CASE WHEN c.is_nullable = 0 THEN ' NOT NULL' ELSE ' NULL' END
                    AS VARCHAR(MAX)), 
                    ', '
                ) + ');' AS CreateTableStatement
            FROM 
                sys.tables t
                INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                INNER JOIN sys.columns c ON t.object_id = c.object_id
                INNER JOIN sys.types tp ON c.user_type_id = tp.user_type_id
            WHERE 
                t.name = '{table}'
            GROUP BY 
                s.name, t.name
            """)
            result = conn.execute(query)
            for row in result:
                ddl_statements.append(row[0])
    
    return ddl_statements

def extract_ddl_generic(engine) -> List[str]:
    """Extract generic DDL statements based on SQLAlchemy inspection"""
    inspector = inspect(engine)
    ddl_statements = []
    
    tables = inspector.get_table_names()
    for table in tables:
        columns = inspector.get_columns(table)
        column_defs = []
        
        for column in columns:
            col_type = str(column['type'])
            nullable = "" if column.get('nullable', True) else " NOT NULL"
            column_def = f"{column['name']} {col_type}{nullable}"
            column_defs.append(column_def)
            
        create_stmt = f"CREATE TABLE {table} (\n  " + ",\n  ".join(column_defs) + "\n);"
        ddl_statements.append(create_stmt)
    
    return ddl_statements

def extract_example_queries(engine, db_type: str) -> List[str]:
    """Extract example queries based on database structure"""
    inspector = inspect(engine)
    example_queries = []
    
    if db_type == 'postgresql':
        schema = 'public'
        tables = inspector.get_table_names(schema=schema)
    else:
        schema = None
        tables = inspector.get_table_names()
    
    for table in tables[:10]:  # Limit to first 10 tables
        try:
            columns = inspector.get_columns(table, schema=schema)
            if not columns:
                continue
                
            table_identifier = f"{schema}.{table}" if schema else table
            
            # Basic SELECT examples
            if db_type == 'mssql':
                example_queries.append(f"SELECT TOP 10 * FROM [{table_identifier}];")
            else:
                example_queries.append(f"SELECT * FROM {table_identifier} LIMIT 10;")
                
            # Column-specific examples
            col_names = [c['name'] for c in columns[:3]]
            if col_names:
                cols_str = ', '.join(col_names)
                if db_type == 'mssql':
                    example_queries.append(f"SELECT TOP 10 {cols_str} FROM [{table_identifier}];")
                else:
                    example_queries.append(f"SELECT {cols_str} FROM {table_identifier} LIMIT 10;")
            
        except Exception as e:
            print(f"Warning: Could not generate examples for table {table}: {e}")
    
    return example_queries

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

# ------------------- Helper Functions ------------------- #

def check_if_model_trained(model_path: Path) -> bool:
    """Check if a model has been trained by looking for ChromaDB files"""
    chroma_file = model_path / "chroma.sqlite3"
    return chroma_file.exists()

def save_connection_params(session_id: str, connection_params: Dict):
    """Save connection parameters for a session"""
    connection_file = CONNECTIONS_DIR / f"session_{session_id}.json"
    safe_params = {k: v for k, v in connection_params.items() if k != 'password'}
    with open(connection_file, 'w') as f:
        json.dump(safe_params, f, indent=2)

def load_connection_params(session_id: str) -> Optional[Dict]:
    """Load connection parameters for a session"""
    connection_file = CONNECTIONS_DIR / f"session_{session_id}.json"
    if connection_file.exists():
        with open(connection_file, 'r') as f:
            return json.load(f)
    return None

def create_vanna_instance(openai_api_key: str, model_path: Path) -> BeastModeVannaTrainer:
    """Create a BEAST MODE Vanna instance with persistent storage"""
    model_path.mkdir(exist_ok=True)
    
    config = {
        'api_key': openai_api_key,
        'model': 'gpt-4o',
        'path': str(model_path)
    }
    
    return BeastModeVannaTrainer(config=config)

# In-memory user session store with persistence backup
user_sessions = {}
session_lock = threading.Lock()

# Load existing sessions on startup
def load_existing_sessions():
    """Load existing sessions from disk"""
    global user_sessions
    try:
        for connection_file in CONNECTIONS_DIR.glob("session_*.json"):
            session_id = connection_file.stem.replace("session_", "")
            
            # Check if model exists for this session
            with open(connection_file, 'r') as f:
                connection_params = json.load(f)
            
            model_id = generate_model_id(connection_params)
            model_path = MODELS_DIR / f"model_{model_id}"
            is_trained = check_if_model_trained(model_path)
            
            # Create a minimal session entry
            user_sessions[session_id] = {
                "model_id": model_id,
                "model_path": model_path,
                "trained": is_trained,
                "connection_params": connection_params,
                "engine": None,  # Will be recreated when needed
                "vn": None,      # Will be loaded when needed
                "args": None     # Will be recreated when needed
            }
            print(f"🔄 Restored session {session_id} - Model trained: {is_trained}")
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing sessions: {e}")

# Load sessions on startup
load_existing_sessions()

# ------------------- Dependency ------------------- #
def get_vanna_session(session_id: str) -> Dict[str, Any]:
    with session_lock:
        session = user_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found. Please connect first.")
        
        # If engine is None, try to recreate it from stored connection params
        if session.get("engine") is None and session.get("connection_params"):
            try:
                # Recreate args object with all required fields
                class Args: pass
                args = Args()
                for field, value in session["connection_params"].items():
                    setattr(args, field, value)
                
                # Ensure password is set to None (will prompt if needed)
                if not hasattr(args, 'password'):
                    args.password = None
                
                # Recreate connection
                conn_string = get_connection_string(args)
                engine = sqlalchemy.create_engine(conn_string)
                
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                # Update session
                session["engine"] = engine
                session["args"] = args
                session["conn_string"] = conn_string
                
                print(f"🔄 Recreated database connection for session {session_id}")
                
            except Exception as e:
                print(f"❌ Failed to recreate connection for session {session_id}: {e}")
                # Instead of raising error, return session as-is for trained model queries
                if session.get("trained") and session.get("vn"):
                    print(f"⚠️  Session has trained model, continuing without database connection")
                    return session
                raise HTTPException(status_code=400, detail=f"Failed to recreate database connection: {e}")
        
        return session

# ------------------- Endpoints ------------------- #
@app.post("/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest):
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
    
    # Generate model ID for persistent storage
    connection_params = req.dict()
    model_id = generate_model_id(connection_params)
    model_path = MODELS_DIR / f"model_{model_id}"
    
    # Save connection parameters
    save_connection_params(session_id, connection_params)
    
    # Check if model is already trained
    is_trained = check_if_model_trained(model_path)
    
    # Store session
    with session_lock:
        user_sessions[session_id] = {
            "args": args,
            "conn_string": conn_string,
            "engine": engine,
            "model_id": model_id,
            "model_path": model_path,
            "vn": None,
            "trained": is_trained,
            "connection_params": connection_params
        }
    
    status_message = f"🔥 Connection established. Model {'already trained' if is_trained else 'ready for BEAST MODE training'}."
    return ConnectResponse(session_id=session_id, message=status_message)

BATCH_SIZE = 50  # Increased for BEAST MODE

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        print(f"🔥 BEAST MODE: Training started for session {req.session_id}")
        session = get_vanna_session(req.session_id)
        print(f"🔥 BEAST MODE: Session retrieved successfully")
        
        engine = session["engine"]
        args = session["args"]
        model_path = session["model_path"]
        
        print(f"🔥 BEAST MODE: Model path: {model_path}")
        print(f"🔥 BEAST MODE: Database type: {args.db_type}")
        
        # Create BEAST MODE Vanna instance with persistent storage
        try:
            print(f"🔥 BEAST MODE: Creating BEAST MODE Vanna instance...")
            vn = create_vanna_instance(req.openai_api_key, model_path)
            session["vn"] = vn
            print(f"🔥 BEAST MODE: BEAST MODE Vanna instance created successfully")
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize BEAST MODE Vanna: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize BEAST MODE Vanna: {e}")
        
        # Check if model is already trained
        if session["trained"]:
            session["vn"] = vn
            print(f"🔥 BEAST MODE: Model already trained, returning early")
            return TrainResponse(message="🔥 BEAST MODE model already trained and loaded from persistent storage.")
        
        # BEAST MODE Training
        try:
            print(f"🔥 STARTING BEAST MODE TRAINING...")
            stats = vn.beast_mode_training(engine, args.db_type, batch_size=BATCH_SIZE)
            
            session["vn"] = vn
            session["trained"] = True
            
            print(f"🎉 BEAST MODE TRAINING COMPLETE!")
            print(f"📊 Total examples trained: {stats.get('total_examples', 0)}")
            
        except Exception as e:
            print(f"❌ ERROR: BEAST MODE training failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"BEAST MODE training failed: {e}")
        
        return TrainResponse(message=f"🔥 BEAST MODE training complete! Trained with {stats.get('total_examples', 0)} examples and saved persistently.")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR in BEAST MODE training endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        session = get_vanna_session(req.session_id)
        
        if not session.get("vn"):
            if not session.get("trained"):
                return QueryResponse(error="Model not trained. Please call BEAST MODE /train first.")
            return QueryResponse(error="Model not loaded. Please reconnect and provide OpenAI API key to reload trained model.")
        
        vn = session["vn"]

        if session.get("engine") is None:
            return QueryResponse(error="If you want to run the SQL query, connect to a database first.")

        # Generate SQL from question
        print(f"🔍 DEBUG: Generating SQL for question: {req.question}")
        sql = vn.ask(req.question)
        print(f"🔍 DEBUG: Generated SQL: {sql}")

        if sql is None:
            return QueryResponse(error="Model could not generate SQL for this question.")



        if session.get("engine") is None:
            return QueryResponse(error="If you want to run the SQL query, connect to a database first.")


        
        if req.return_sql_only:
            return QueryResponse(sql=sql)
        
        # Run SQL and return results
        engine = session["engine"]
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(sql))
            rows = result.fetchall()
            columns = list(result.keys())  # Convert to list
            answer = [dict(zip(columns, row)) for row in rows]
        
        print(f"🔍 DEBUG: Query executed successfully, {len(answer)} rows returned")
        return QueryResponse(sql=sql, answer=answer)
        
    except Exception as e:
        print(f"❌ ERROR in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return QueryResponse(error=str(e))

@app.post("/reload-model")
def reload_model(req: TrainRequest):
    """Reload a trained BEAST MODE model from persistent storage"""
    session = get_vanna_session(req.session_id)
    model_path = session["model_path"]
    
    if not check_if_model_trained(model_path):
        raise HTTPException(status_code=400, detail="No trained BEAST MODE model found. Please train first.")
    
    try:
        vn = create_vanna_instance(req.openai_api_key, model_path)
        session["vn"] = vn
        session["trained"] = True
        return {"message": "🔥 BEAST MODE model reloaded successfully from persistent storage."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload BEAST MODE model: {e}")

@app.post("/add-question", response_model=AddQuestionResponse)
def add_question(req: AddQuestionRequest):
    session = get_vanna_session(req.session_id)
    if not session.get("trained") or not session.get("vn"):
        raise HTTPException(status_code=400, detail="BEAST MODE model not trained. Please call /train first.")
    vn = session["vn"]
    try:
        sql = vn.ask(req.question)
        vn.train(question=req.question, sql=sql)
        return AddQuestionResponse(message="🔥 Question added to BEAST MODE RAG vector store and saved persistently.", sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question to BEAST MODE: {e}")

@app.post("/agent-query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest):
    """🔥 BEAST MODE Agentic, iterative query refinement with error analysis."""
    session = get_vanna_session(req.session_id)
    if not session.get("trained") or not session.get("vn"):
        return AgentQueryResponse(error="BEAST MODE model not trained. Please call /train first.", attempts=0)
    
    vn = session["vn"]
    engine = session.get("engine")
    if engine is None:
        return AgentQueryResponse(error="If you want to run the SQL query, connect to a database first.", attempts=0)
    context = ""
    sql = None
    answer = None
    analysis = None
    last_error = None
    
    for attempt in range(1, req.max_attempts + 1):
        prompt = req.question if not context else f"{req.question}\n\nError encountered: {context}"
        try:
            sql = vn.ask(prompt)
            if sql is None:
                raise ValueError("Model returned no SQL")
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql))
                rows = result.fetchall()
                columns = result.keys()
                answer = [dict(zip(columns, row)) for row in rows]
            
            analysis_prompt = f"Given the following data (columns: {columns}, rows: {answer[:5]}), draft actionable steps for a developer to implement based on the user's question: '{req.question}'."
            analysis = vn.ask(analysis_prompt)
            return AgentQueryResponse(sql=sql, answer=answer, analysis=analysis, attempts=attempt)
        except Exception as e:
            context = str(e)
            last_error = context
    
    return AgentQueryResponse(error=f"🔥 BEAST MODE failed after {req.max_attempts} attempts. Last error: {last_error}", attempts=req.max_attempts)

@app.get("/training-status/{session_id}")
def get_training_status(session_id: str):
    """Get BEAST MODE training status for a session"""
    try:
        session = get_vanna_session(session_id)
        model_path = session["model_path"]
        is_trained = check_if_model_trained(model_path)
        
        training_data_count = 0
        if session.get("vn"):
            try:
                training_data = session["vn"].get_training_data()
                training_data_count = len(training_data) if training_data is not None else 0
            except:
                pass
        
        return {
            "session_id": session_id,
            "model_id": session["model_id"],
            "trained": is_trained,
            "model_loaded": session.get("vn") is not None,
            "training_data_count": training_data_count,
            "model_path": str(model_path),
            "beast_mode": True
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/restore-session/{session_id}")
def restore_session(session_id: str):
    """Restore a BEAST MODE session from saved connection parameters"""
    try:
        connection_file = CONNECTIONS_DIR / f"session_{session_id}.json"
        if not connection_file.exists():
            raise HTTPException(status_code=404, detail="Session connection file not found")
        
        # Load connection parameters
        with open(connection_file, 'r') as f:
            connection_params = json.load(f)
        
        # Generate model info
        model_id = generate_model_id(connection_params)
        model_path = MODELS_DIR / f"model_{model_id}"
        is_trained = check_if_model_trained(model_path)
        
        # Create session entry (engine will be created when needed)
        with session_lock:
            user_sessions[session_id] = {
                "model_id": model_id,
                "model_path": model_path,
                "trained": is_trained,
                "connection_params": connection_params,
                "engine": None,  # Will be recreated when needed
                "vn": None,      # Will be loaded when needed
                "args": None     # Will be recreated when needed
            }
        
        return {
            "message": f"🔥 BEAST MODE session {session_id} restored successfully",
            "trained": is_trained,
            "model_id": model_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore BEAST MODE session: {e}")

@app.get("/list-sessions")
def list_sessions():
    """List all available BEAST MODE sessions"""
    sessions = []
    try:
        for connection_file in CONNECTIONS_DIR.glob("session_*.json"):
            session_id = connection_file.stem.replace("session_", "")
            
            with open(connection_file, 'r') as f:
                connection_params = json.load(f)
            
            model_id = generate_model_id(connection_params)
            model_path = MODELS_DIR / f"model_{model_id}"
            is_trained = check_if_model_trained(model_path)
            
            sessions.append({
                "session_id": session_id,
                "model_id": model_id,
                "trained": is_trained,
                "db_type": connection_params.get("db_type"),
                "host": connection_params.get("host"),
                "dbname": connection_params.get("dbname"),
                "beast_mode": True
            })
            
        return {"sessions": sessions}
        
    except Exception as e:
        return {"sessions": [], "error": str(e)}

@app.get("/")
def root():
    return {
        "message": "🔥 Welcome to the BEAST MODE Vanna NLP-to-SQL API with 500+ Training Examples! See /docs for OpenAPI documentation.",
        "beast_mode": True,
        "features": [
            "🔥 BEAST MODE training with 500+ examples",
            "📊 Advanced query patterns (JOINs, subqueries, window functions)",
            "💾 Persistent model training data",
            "🔄 Automatic model reloading", 
            "📋 Session management",
            "🗂️ ChromaDB vector storage",
            "🌐 Multi-database support",
            "🤖 Agentic query refinement",
            "📚 Business logic integration"
        ],
        "training_examples": "500+",
        "accuracy": "BEAST MODE",
        "version": "2.0.0"
    }