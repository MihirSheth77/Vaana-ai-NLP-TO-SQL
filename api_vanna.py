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

# Import OpenAI Agents SDK
try:
    from agents import Agent, Runner
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False
    print("⚠️  OpenAI Agents SDK not available. Install with: pip install openai-agents")

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

def normalize_sql_output(output: Any) -> Optional[str]:
    """Simple extraction using modern LLM structured output"""
    if output is None:
        return None
    
    if isinstance(output, str):
        # Modern approach: extract from structured tags
        if '<sql>' in output and '</sql>' in output:
            try:
                sql_content = output.split('<sql>')[1].split('</sql>')[0].strip()
                if sql_content and sql_content != 'NO_SQL_POSSIBLE':
                    return sql_content
            except IndexError:
                pass
        
        # Legacy fallback for existing trained models
        if "Extracted SQL: " in output:
            return output.split("Extracted SQL: ", 1)[1].strip()
        
        # Extract SQL from markdown code blocks
        import re
        sql_match = re.search(r'```sql\s*\n(.*?)\n```', output, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Fallback: if it's already clean SQL, return it
        if any(keyword in output.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']):
            # Find the line that starts with SQL
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']):
                    return line
        
        return output.strip()
    
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, str):
            return normalize_sql_output(first)
    
    return None

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
    print(f"[DEBUG] /connect called. New session_id: {session_id}")
    
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
        print(f"[DEBUG] /connect: user_sessions keys after connect: {list(user_sessions.keys())}")
    
    status_message = f"🔥 Connection established. Model {'already trained' if is_trained else 'ready for BEAST MODE training'}."
    return ConnectResponse(session_id=session_id, message=status_message)

BATCH_SIZE = 50  # Increased for BEAST MODE

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        # Debug: Print the received API key (mask all but last 4 chars)
        masked_key = req.openai_api_key[:6] + "..." + req.openai_api_key[-4:] if req.openai_api_key else None
        print(f"[DEBUG] Received OpenAI API key: {masked_key}")
        # Strip whitespace from API key
        req.openai_api_key = req.openai_api_key.strip() if req.openai_api_key else req.openai_api_key
        print(f"[DEBUG] Stripped OpenAI API key: {masked_key}")
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
    print(f"[DEBUG] /query called. session_id from request: {req.session_id}")
    print(f"[DEBUG] /query: user_sessions keys at query: {list(user_sessions.keys())}")
    try:
        session = get_vanna_session(req.session_id)
        print(f"[DEBUG] Session at query: {session}")
        print(f"[DEBUG] Engine at query: {session.get('engine')}")
        
        if not session.get("vn"):
            if not session.get("trained"):
                return QueryResponse(error="Model not trained. Please call BEAST MODE /train first.")
            return QueryResponse(error="Model not loaded. Please reconnect and provide OpenAI API key to reload trained model.")
        
        vn = session["vn"]
        engine = session.get("engine")
        
        # Generate SQL from question
        print(f"🔍 DEBUG: Generating SQL for question: {req.question}")
        sql = vn.ask(req.question)  # Use directly without normalize_sql_output
        print(f"🔍 DEBUG: Generated SQL: {sql}")
        
        if sql is None or not sql.strip():
            return QueryResponse(error="Model could not generate SQL for this question.")
        
        # Return SQL only if requested
        if req.return_sql_only:
            return QueryResponse(sql=sql)
        
        # Check if we have database connection for execution
        if engine is None:
            return QueryResponse(error="Database connection not available. Please reconnect to database.")
        
        # Execute SQL and return results
        try:
            print(f"🔍 DEBUG: Executing SQL on database...")
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql))
                rows = result.fetchall()
                columns = list(result.keys())  # Convert to list
                answer = [dict(zip(columns, row)) for row in rows]
            
            print(f"🔍 DEBUG: Query executed successfully, {len(answer)} rows returned")
            return QueryResponse(sql=sql, answer=answer)
            
        except Exception as sql_error:
            # Log the ACTUAL SQL execution error
            print(f"❌ SQL EXECUTION ERROR: {sql_error}")
            print(f"❌ Failed SQL: {sql}")
            import traceback
            traceback.print_exc()
            
            # Return the real error to help debugging
            return QueryResponse(
                sql=sql,
                error=f"SQL execution failed: {str(sql_error)}"
            )
            
    except Exception as e:
        print(f"❌ ERROR in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return QueryResponse(error=str(e))

@app.post("/reload-model")
def reload_model(req: TrainRequest):
    """Reload a trained BEAST MODE model from persistent storage"""
    # Debug: Print the received API key (mask all but last 4 chars)
    masked_key = req.openai_api_key[:6] + "..." + req.openai_api_key[-4:] if req.openai_api_key else None
    print(f"[DEBUG] Received OpenAI API key: {masked_key}")
    # Strip whitespace from API key
    req.openai_api_key = req.openai_api_key.strip() if req.openai_api_key else req.openai_api_key
    print(f"[DEBUG] Stripped OpenAI API key: {masked_key}")
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
        sql_raw = vn.ask(req.question)
        sql = normalize_sql_output(sql_raw)
        if sql is None:
            raise ValueError("Model could not generate SQL for this question.")
        vn.train(question=req.question, sql=sql)
        return AddQuestionResponse(message="🔥 Question added to BEAST MODE RAG vector store and saved persistently.", sql=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question to BEAST MODE: {e}")

@app.post("/agent-query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest):
    print(f"[DEBUG] /agent-query called. session_id from request: {req.session_id}")
    print(f"[DEBUG] /agent-query: user_sessions keys at agent-query: {list(user_sessions.keys())}")
    session = get_vanna_session(req.session_id)
    print(f"[DEBUG] Session at agent-query: {session}")
    print(f"[DEBUG] Engine at agent-query: {session.get('engine')}")
    
    if not session.get("trained") or not session.get("vn"):
        return AgentQueryResponse(error="BEAST MODE model not trained. Please call /train first.", attempts=0)
    
    vn = session["vn"]
    engine = session.get("engine")
    
    # Get database schema for context
    schema_info = ""
    if engine:
        try:
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            schema_parts = []
            for table in tables[:10]:  # Limit to avoid token overflow
                columns = inspector.get_columns(table)
                col_names = [col['name'] for col in columns]
                schema_parts.append(f"Table {table}: {', '.join(col_names)}")
            schema_info = "\n".join(schema_parts)
        except Exception as e:
            print(f"Could not extract schema: {e}")
            schema_info = "Schema information unavailable"
    
    conversation_history = []
    
    for attempt in range(1, req.max_attempts + 1):
        try:
            print(f"🔄 Attempt {attempt}/{req.max_attempts}")
            
            # Build iterative prompt based on previous attempts
            if attempt == 1:
                # Initial prompt
                system_prompt = f"""You are an expert SQL query generator. Generate ONLY valid SQL queries based on the user's natural language request.

Database Schema:
{schema_info}

Rules:
1. Return ONLY the SQL query, no explanations
2. Use proper PostgreSQL syntax
3. Always use proper table and column names from the schema
4. Handle NULL values appropriately
5. Use appropriate JOINs when needed
6. Include proper WHERE clauses for filtering
7. Use aggregation functions correctly (SUM, COUNT, AVG, etc.)

Return only the SQL query without any markdown formatting or additional text."""

                user_prompt = f"Generate a SQL query for: {req.question}"
                
            else:
                # Iterative refinement prompt with error feedback
                error_context = conversation_history[-1] if conversation_history else ""
                
                system_prompt = f"""You are an expert SQL query generator. The previous SQL query failed with an error. 
Please analyze the error and generate a corrected SQL query.

Database Schema:
{schema_info}

Previous attempt failed with this error:
{error_context}

CRITICAL RULES:
1. Return ONLY the complete, corrected SQL query
2. If the error shows "syntax error at end of input", ensure you complete the entire query structure
3. For window functions, always close all parentheses: ) AS column_name
4. For CTEs, ensure all parts are complete: SELECT ... FROM cte_name
5. Always end with a semicolon
6. Use proper PostgreSQL syntax
7. Fix the specific syntax error mentioned above

The previous query appears to be truncated. Generate the COMPLETE corrected query."""

                user_prompt = f"Original question: {req.question}\n\nPlease provide a corrected SQL query that fixes the previous error."
            
            # Use OpenAI SDK directly for better control
            import openai
            client = openai.OpenAI(api_key=vn.config.get('api_key'))
            
            response = client.chat.completions.create(
                model=vn.config.get('model', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent SQL generation
                max_tokens=1500,  # Increased from 500 to handle complex queries
                top_p=0.9
            )
            
            # Extract SQL from response
            sql = response.choices[0].message.content.strip()
            
            # Clean up the SQL (remove markdown if present)
            import re
            if '```sql' in sql.lower():
                sql_match = re.search(r'```sql\s*\n(.*?)\n```', sql, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    sql = sql_match.group(1).strip()
            elif '```' in sql:
                sql_match = re.search(r'```\s*\n(.*?)\n```', sql, re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1).strip()
            
            # Ensure SQL ends with semicolon if it's a complete query
            if sql and not sql.strip().endswith(';') and any(sql.upper().strip().startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                sql = sql.strip() + ';'
            
            print(f"🔍 DEBUG: Attempt {attempt} - Generated SQL: {sql}")
            
            if not sql or not sql.strip():
                error_msg = f"No SQL generated on attempt {attempt}"
                print(f"❌ {error_msg}")
                conversation_history.append(f"Attempt {attempt}: {error_msg}")
                continue
            
            # Try to execute the SQL if we have a database connection
            answer = None
            analysis = f"Generated SQL query on attempt {attempt}"
            
            if engine:
                try:
                    print(f"🔍 DEBUG: Attempting to execute SQL: {sql}")
                    with engine.connect() as conn:
                        result = conn.execute(sqlalchemy.text(sql))
                        rows = result.fetchall()
                        columns = list(result.keys())
                        answer = [dict(zip(columns, row)) for row in rows]
                    
                    analysis = f"✅ Successfully executed SQL query on attempt {attempt}. Retrieved {len(answer)} rows."
                    print(f"✅ Success on attempt {attempt}: {len(answer)} rows returned")
                    
                    # Success! Return the result
                    return AgentQueryResponse(sql=sql, answer=answer, analysis=analysis, attempts=attempt)
                    
                except Exception as sql_error:
                    error_msg = str(sql_error)
                    print(f"❌ SQL execution error on attempt {attempt}: {error_msg}")
                    
                    # Add detailed error context for next iteration
                    error_context = f"""
SQL Query: {sql}
Error: {error_msg}
Error Type: {type(sql_error).__name__}
Query Length: {len(sql)} characters
Issue: {"Query appears truncated - missing closing parentheses or SELECT clause" if "syntax error at end of input" in error_msg else "SQL execution error"}
"""
                    conversation_history.append(error_context)
                    analysis = f"Attempt {attempt}: SQL execution failed - {error_msg}"
                    
                    # If this is the last attempt, return with error
                    if attempt >= req.max_attempts:
                        return AgentQueryResponse(
                            sql=sql, 
                            error=f"SQL execution failed after {req.max_attempts} attempts. Final error: {error_msg}", 
                            analysis=analysis, 
                            attempts=attempt
                        )
                    
                    # Continue to next iteration with error feedback
                    continue
            else:
                # No database connection, but SQL was generated successfully
                analysis = f"Generated SQL on attempt {attempt}, but no database connection available for execution"
            return AgentQueryResponse(sql=sql, answer=answer, analysis=analysis, attempts=attempt)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error on attempt {attempt}: {error_msg}")
            conversation_history.append(f"Attempt {attempt} - Generation Error: {error_msg}")
            
            if attempt >= req.max_attempts:
                return AgentQueryResponse(error=f"Failed after {req.max_attempts} attempts: {error_msg}", attempts=attempt)
    
    return AgentQueryResponse(error=f"Failed to generate valid SQL after {req.max_attempts} attempts", attempts=req.max_attempts)

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
    agents_status = "✅ Available" if AGENTS_SDK_AVAILABLE else "❌ Not installed (pip install openai-agents)"
    
    return {
        "message": "🔥 Welcome to the BEAST MODE Vanna NLP-to-SQL API with OpenAI Agents SDK Integration!",
        "beast_mode": True,
        "agents_sdk": agents_status,
        "endpoints": {
            "legacy_agent": "/agent-query (manual retry loop)",
            "modern_agent": "/agents/query (OpenAI Agents SDK)", 
            "simple_query": "/query (single attempt)"
        },
        "features": [
            "🔥 BEAST MODE training with 500+ examples",
            "🤖 OpenAI Agents SDK integration",
            "📊 Advanced query patterns (JOINs, subqueries, window functions)",
            "💾 Persistent model training data",
            "🔄 Automatic model reloading", 
            "📋 Session management",
            "🗂️ ChromaDB vector storage",
            "🌐 Multi-database support",
            "🔧 Built-in SQL execution tools",
            "📚 Business logic integration",
            "👁️ Agent tracing and visualization"
        ],
        "training_examples": "500+",
        "accuracy": "BEAST MODE",
        "version": "2.0.0"
    }

@app.get("/list-models")
def list_models():
    """List all available trained models"""
    models = []
    try:
        for model_dir in MODELS_DIR.glob("model_*"):
            if model_dir.is_dir():
                model_id = model_dir.name.replace("model_", "")
                is_trained = check_if_model_trained(model_dir)
                models.append({
                    "model_id": model_id,
                    "path": str(model_dir),
                    "trained": is_trained
                })
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}

# ------------------- Agent Tools for OpenAI Agents SDK ------------------- #

def create_sql_execution_tool(engine, schema_info: str):
    """Create a SQL execution tool for the Agents SDK"""
    def execute_sql(sql_query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return the results.
        
        Args:
            sql_query: The SQL query to execute
            
        Returns:
            Dictionary with success status, results, or error message
        """
        try:
            if not sql_query or not sql_query.strip():
                return {"success": False, "error": "Empty SQL query provided"}
            
            print(f"🔍 Executing SQL: {sql_query}")
            with engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql_query))
                rows = result.fetchall()
                columns = list(result.keys())
                data = [dict(zip(columns, row)) for row in rows]
            
            return {
                "success": True,
                "row_count": len(data),
                "data": data[:10],  # Limit to first 10 rows for response size
                "total_rows": len(data)
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ SQL execution error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__
            }
    
    return execute_sql

def create_schema_inspection_tool(schema_info: str):
    """Create a schema inspection tool for the Agents SDK"""
    def get_database_schema() -> str:
        """
        Get the database schema information.
        
        Returns:
            String containing database schema details
        """
        return f"Database Schema:\n{schema_info}"
    
    return get_database_schema

@app.post("/agents/query", response_model=AgentQueryResponse)
def agents_query(req: AgentQueryRequest):
    """Enhanced agent-based query using OpenAI Agents SDK"""
    print(f"[DEBUG] /agents/query called. session_id: {req.session_id}")
    
    if not AGENTS_SDK_AVAILABLE:
        return AgentQueryResponse(
            error="OpenAI Agents SDK not available. Install with: pip install openai-agents", 
            attempts=0
        )
    
    try:
        session = get_vanna_session(req.session_id)
        
        if not session.get("trained") or not session.get("vn"):
            return AgentQueryResponse(
                error="BEAST MODE model not trained. Please call /train first.", 
                attempts=0
            )
        
        vn = session["vn"]
        engine = session.get("engine")
        
        # Get database schema for context
        schema_info = ""
        if engine:
            try:
                from sqlalchemy import inspect
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                schema_parts = []
                for table in tables[:10]:  # Limit to avoid token overflow
                    columns = inspector.get_columns(table)
                    col_names = [col['name'] for col in columns]
                    schema_parts.append(f"Table {table}: {', '.join(col_names)}")
                schema_info = "\n".join(schema_parts)
            except Exception as e:
                print(f"Could not extract schema: {e}")
                schema_info = "Schema information unavailable"
        
        # Create tools for the agent
        tools = []
        if engine:
            tools.append(create_sql_execution_tool(engine, schema_info))
        tools.append(create_schema_inspection_tool(schema_info))
        
        # Create the SQL Expert Agent
        sql_agent = Agent(
            name="SQLExpert",
            instructions=f"""You are an expert SQL query generator and database analyst. Your goal is to:

1. Generate accurate PostgreSQL queries from natural language questions
2. Execute queries to validate they work correctly  
3. If a query fails, analyze the error and fix it
4. Provide helpful analysis of the results

Database Schema:
{schema_info}

CRITICAL RULES:
- Always generate complete, valid PostgreSQL syntax
- Use proper table and column names from the schema
- Handle NULL values appropriately with NULLIF() where needed
- For window functions, ensure all parentheses are properly closed
- For CTEs, ensure all parts are complete with proper SELECT statements
- Test your SQL by executing it first
- If execution fails, fix the syntax errors and try again
- Provide clear analysis of what the query does and what the results mean

Always execute your SQL queries to ensure they work before providing the final answer.""",
            tools=tools
        )
        
        # Run the agent with the user's question
        print(f"🤖 Running SQL Agent for question: {req.question}")
        
        result = Runner.run_sync(sql_agent, req.question)
        
        # Parse the result
        final_output = result.final_output
        
        # Try to extract SQL from the agent's response
        sql = None
        if hasattr(result, 'messages'):
            for message in result.messages:
                if hasattr(message, 'tool_calls'):
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == 'execute_sql':
                            sql = json.loads(tool_call.function.arguments).get('sql_query')
                            break
        
        # Get the last successful execution result
        answer = None
        execution_success = False
        for message in reversed(result.messages) if hasattr(result, 'messages') else []:
            if hasattr(message, 'tool_calls'):
                for tool_call in message.tool_calls:
                    if tool_call.function.name == 'execute_sql':
                        try:
                            args = json.loads(tool_call.function.arguments)
                            sql = args.get('sql_query')
                            # Check if this execution was successful by looking at the next message
                            # (This is a simplified approach - in practice you'd want better result tracking)
                            execution_success = True
                            break
                        except:
                            pass
        
        # Count attempts (simplified - the SDK handles retries internally)
        attempts = 1
        if hasattr(result, 'messages'):
            tool_calls = sum(1 for msg in result.messages 
                           if hasattr(msg, 'tool_calls') and 
                           any(tc.function.name == 'execute_sql' for tc in msg.tool_calls))
            attempts = max(1, tool_calls)
        
        return AgentQueryResponse(
            sql=sql,
            answer=answer,
            analysis=final_output,
            attempts=attempts
        )
        
    except Exception as e:
        print(f"❌ Error in agents query endpoint: {e}")
        import traceback
        traceback.print_exc()
        return AgentQueryResponse(
            error=f"Agent execution failed: {str(e)}", 
            attempts=1
        )

# ------------------- Server Startup ------------------- #
if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting BEAST MODE Vanna API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)