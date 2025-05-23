#!/usr/bin/env python3
"""
Automatic Vanna.AI training script for any database
This script connects to any supported database, extracts schema information,
and trains a Vanna.AI model on the database structure.
"""

import os
import argparse
import sqlalchemy
import getpass
import json
import datetime
from sqlalchemy import inspect, text
from typing import Dict, List, Optional, Tuple, Any

# Import Vanna modules
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

# Global variables to maintain database connection
g_engine = None
g_connection_string = None
g_connection_info = {}

# Directory for storing successful query history
QUERY_HISTORY_DIR = "query_history"
# Directory for business terminology
TERMINOLOGY_DIR = "business_terminology"
# Directory for configuration
CONFIG_DIR = "config"

class VannaTrainer(ChromaDB_VectorStore, OpenAI_Chat):
    """Custom Vanna implementation using ChromaDB and OpenAI"""
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self.engine = None
        
    def maintain_connection(self, engine):
        """Store the engine connection to maintain it between calls"""
        global g_engine, g_connection_string
        print("Setting up database connection for later use")
        self.engine = engine
        g_engine = engine
        return self.engine
        
    def execute_query(self, sql):
        """Execute the query using the stored connection"""
        global g_engine, g_connection_string
        
        print("Attempting to execute query with available connection...")
        
        # First try with the instance engine
        if hasattr(self, 'engine') and self.engine:
            try:
                print("Using instance engine connection")
                with self.engine.connect() as conn:
                    return conn.execute(text(sql))
            except Exception as e:
                print(f"Connection error with instance engine: {e}. Attempting to reconnect...")
                connection_string = get_stored_connection_string()
                if connection_string:
                    try:
                        print(f"Reconnecting with stored connection string...")
                        self.engine = sqlalchemy.create_engine(connection_string, pool_pre_ping=True, pool_recycle=3600)
                        g_engine = self.engine
                        with self.engine.connect() as conn:
                            return conn.execute(text(sql))
                    except Exception as reconnect_error:
                        print(f"Reconnection failed: {reconnect_error}")
                        raise ConnectionError(f"Failed to reconnect to the database: {reconnect_error}")
                else:
                    raise ConnectionError("Cannot reconnect, connection string not available")
        # Try using global engine if instance engine is not available
        elif g_engine:
            try:
                print("Using global engine connection")
                with g_engine.connect() as conn:
                    return conn.execute(text(sql))
            except Exception as e:
                print(f"Connection error with global engine: {e}")
                connection_string = get_stored_connection_string()
                if connection_string:
                    try:
                        print("Reconnecting with stored global connection string...")
                        g_engine = sqlalchemy.create_engine(connection_string, pool_pre_ping=True, pool_recycle=3600)
                        self.engine = g_engine  # Also update instance engine
                        with g_engine.connect() as conn:
                            return conn.execute(text(sql))
                    except Exception as reconnect_error:
                        print(f"Global reconnection failed: {reconnect_error}")
                        raise ConnectionError(f"Failed to reconnect to the database: {reconnect_error}")
            
        # If we get here, no connection is available
        connection_string = get_stored_connection_string()
        if connection_string:
            print("No active connection found, but connection string is available. Creating new connection...")
            try:
                new_engine = sqlalchemy.create_engine(connection_string, pool_pre_ping=True, pool_recycle=3600)
                self.engine = new_engine
                g_engine = new_engine
                
                with new_engine.connect() as conn:
                    return conn.execute(text(sql))
            except Exception as fresh_conn_error:
                raise ConnectionError(f"Failed to create a fresh connection: {fresh_conn_error}")
        else:
            raise ConnectionError("Database connection not available and no connection string to reconnect. Please reconnect to the database.")
    
    def save_successful_query(self, question, sql, feedback_score=None):
        """Save a successful query for retraining"""
        # Create directory if it doesn't exist
        os.makedirs(QUERY_HISTORY_DIR, exist_ok=True)
        
        # Create a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(QUERY_HISTORY_DIR, f"query_{timestamp}.json")
        
        # Save the query details
        query_data = {
            "question": question,
            "sql": sql,
            "timestamp": timestamp,
            "feedback_score": feedback_score
        }
        
        with open(filename, "w") as f:
            json.dump(query_data, f, indent=2)
        
        print(f"Query saved to {filename}")
        return filename
    
    def retrain_from_history(self, min_feedback_score=None):
        """Retrain the model using saved successful queries"""
        if not os.path.exists(QUERY_HISTORY_DIR):
            print("No query history found.")
            return 0
        
        # Get all query history files
        files = os.listdir(QUERY_HISTORY_DIR)
        files = [f for f in files if f.endswith('.json')]
        
        count = 0
        for file in files:
            try:
                with open(os.path.join(QUERY_HISTORY_DIR, file), 'r') as f:
                    query_data = json.load(f)
                
                # Skip if below minimum feedback score
                if min_feedback_score is not None and query_data.get('feedback_score') is not None:
                    if query_data['feedback_score'] < min_feedback_score:
                        continue
                
                # Train with the question-SQL pair
                if 'question' in query_data and 'sql' in query_data:
                    self.train(question=query_data['question'], sql=query_data['sql'])
                    count += 1
                    print(".", end="", flush=True)
            except Exception as e:
                print(f"\nError retraining with {file}: {e}")
        
        print(f"\nRetrained with {count} successful queries from history.")
        return count

def store_connection_info(conn_string, db_type, host=None, port=None, dbname=None, username=None):
    """Store connection info globally to maintain it between sessions"""
    global g_connection_string, g_connection_info
    g_connection_string = conn_string
    g_connection_info = {
        "db_type": db_type,
        "connection_string": conn_string,
        "host": host,
        "port": port,
        "dbname": dbname,
        "username": username
    }
    # Also save to a temp file for absolute recovery
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_DIR, ".current_connection.json"), "w") as f:
        json.dump(g_connection_info, f)
    return conn_string

def get_stored_connection_string():
    """Get the stored connection string, including recovery from file if needed"""
    global g_connection_string, g_connection_info
    # First try the global variable
    if g_connection_string:
        return g_connection_string
    
    # If not available, try to recover from temp file
    try:
        if os.path.exists(os.path.join(CONFIG_DIR, ".current_connection.json")):
            with open(os.path.join(CONFIG_DIR, ".current_connection.json"), "r") as f:
                g_connection_info = json.load(f)
                g_connection_string = g_connection_info.get("connection_string")
                return g_connection_string
    except:
        pass
    
    return None

def get_connection_string(args) -> str:
    """Build a connection string from provided arguments"""
    password = args.password or getpass.getpass("Database password: ")
    
    if args.connection_string:
        # User provided full connection string
        return args.connection_string
    
    # Common connection parameters for various database types
    common_params = "?connect_timeout=30&application_name=vanna_ai"
    
    # Build connection string based on database type
    if args.db_type == 'postgresql':
        # For PostgreSQL, add params for better connection reliability
        pg_params = f"{common_params}&client_encoding=utf8&keepalives=1&keepalives_idle=30&keepalives_interval=10"
        return f"postgresql://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}{pg_params}"
    elif args.db_type == 'mysql':
        mysql_params = f"{common_params}&charset=utf8mb4&autocommit=true&use_unicode=1"
        return f"mysql+pymysql://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}{mysql_params}"
    elif args.db_type == 'mssql':
        mssql_params = f"{common_params}&timeout=30"
        return f"mssql+pyodbc://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}?driver=ODBC+Driver+17+for+SQL+Server{mssql_params}"
    elif args.db_type == 'oracle':
        oracle_params = f"{common_params}"
        return f"oracle+cx_oracle://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}{oracle_params}"
    elif args.db_type == 'sqlite':
        return f"sqlite:///{args.dbname}"
    elif args.db_type == 'snowflake':
        account = args.host.split('.')[0] if '.' in args.host else args.host
        snowflake_params = f"{common_params}"
        return f"snowflake://{args.username}:{password}@{account}/{args.dbname}{snowflake_params}"
    elif args.db_type == 'bigquery':
        # For BigQuery, we'll use the credentials file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials_file
        return f"bigquery://{args.project_id}/{args.dbname}"
    elif args.db_type == 'redshift':
        redshift_params = f"{common_params}&sslmode=prefer"
        return f"redshift+psycopg2://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}{redshift_params}"
    elif args.db_type == 'duckdb':
        return f"duckdb:///{args.dbname}"
    else:
        raise ValueError(f"Unsupported database type: {args.db_type}")

def extract_ddl_postgres(engine) -> List[str]:
    """Extract DDL statements from PostgreSQL"""
    inspector = inspect(engine)
    ddl_statements = []
    
    # Get all schemas
    schemas = inspector.get_schema_names()
    for schema in schemas:
        if schema in ('pg_catalog', 'information_schema'):
            continue
            
        # Get all tables in the schema
        tables = inspector.get_table_names(schema=schema)
        for table in tables:
            # Get table DDL
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
            
            # Get column comments
            columns = inspector.get_columns(table, schema=schema)
            for column in columns:
                ddl_statements.append(f"COMMENT ON COLUMN {schema}.{table}.{column['name']} IS 'Column {column['name']} with type {column['type']}'")
    
    return ddl_statements

def extract_ddl_mysql(engine) -> List[str]:
    """Extract DDL statements from MySQL"""
    inspector = inspect(engine)
    ddl_statements = []
    
    # Get all tables
    tables = inspector.get_table_names()
    for table in tables:
        # Get DDL
        with engine.connect() as conn:
            query = text(f"SHOW CREATE TABLE `{table}`")
            result = conn.execute(query)
            for row in result:
                ddl_statements.append(row[1])  # Second column has CREATE TABLE statement
    
    return ddl_statements

def extract_ddl_mssql(engine) -> List[str]:
    """Extract DDL statements from MS SQL Server"""
    inspector = inspect(engine)
    ddl_statements = []
    
    # Get all tables
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
    
    # Get tables (ignoring schema for simplicity)
    tables = inspector.get_table_names()
    
    for table in tables:
        # Build CREATE TABLE statement
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
    """Extract example queries based on database structure with real values"""
    inspector = inspect(engine)
    example_queries = []
    
    # Get tables
    if db_type == 'postgresql':
        schema = 'public'
        tables = inspector.get_table_names(schema=schema)
    else:
        schema = None
        tables = inspector.get_table_names()
    
    for table in tables:
        # Get columns
        try:
            columns = inspector.get_columns(table, schema=schema)
            if not columns:
                continue
                
            # Basic query patterns applicable to most databases
            table_identifier = f"{schema}.{table}" if schema else table
            
            # Sample real values from this table for more useful examples
            real_values = {}
            try:
                # Get a sample row to extract real values
                with engine.connect() as conn:
                    if db_type == 'mssql':
                        query = text(f"SELECT TOP 1 * FROM [{table_identifier}]")
                    else:
                        query = text(f"SELECT * FROM {table_identifier} LIMIT 1")
                    
                    result = conn.execute(query)
                    sample_row = result.fetchone()
                    
                    if sample_row is not None:
                        # Extract column names and values
                        col_names = result.keys()
                        for i, col_name in enumerate(col_names):
                            # Only store string and numeric values that aren't NULL
                            if sample_row[i] is not None:
                                value = sample_row[i]
                                # Format based on type
                                if isinstance(value, str):
                                    real_values[col_name] = f"'{value}'"
                                elif isinstance(value, (int, float)):
                                    real_values[col_name] = str(value)
                                # Skip other types like dates, complex objects, etc.
            except Exception as e:
                print(f"Warning: Could not sample values from table {table}: {e}")
            
            # SELECT all columns
            if db_type == 'mssql':
                example_queries.append(f"SELECT TOP 10 * FROM [{table_identifier}];")
            else:
                example_queries.append(f"SELECT * FROM {table_identifier} LIMIT 10;")
                
            # SELECT specific columns
            col_names = [c['name'] for c in columns[:3]]  # First 3 columns
            if col_names:
                if db_type == 'mssql':
                    cols_formatted = ', '.join([f"[{c}]" for c in col_names])
                    example_queries.append(f"SELECT TOP 10 {cols_formatted} FROM [{table_identifier}];")
                else:
                    cols_formatted = ', '.join(col_names)
                    example_queries.append(f"SELECT {cols_formatted} FROM {table_identifier} LIMIT 10;")
            
            # WHERE clause for a column with a real value
            if columns and len(columns) > 0:
                # Try to find a suitable column for WHERE clause
                for col in columns:
                    col_name = col['name']
                    # Use a real value if available
                    if col_name in real_values:
                        value = real_values[col_name]
                if db_type == 'mssql':
                            example_queries.append(f"SELECT TOP 10 * FROM [{table_identifier}] WHERE [{col_name}] = {value};")
                        else:
                            example_queries.append(f"SELECT * FROM {table_identifier} WHERE {col_name} = {value} LIMIT 10;")
                        break
                else:
                    # Fallback to first column with placeholder if no suitable real value found
                    col = columns[0]['name']
                    if db_type == 'mssql':
                        example_queries.append(f"SELECT TOP 10 * FROM [{table_identifier}] WHERE [{col}] = 1;")
                    else:
                        example_queries.append(f"SELECT * FROM {table_identifier} WHERE {col} = 1 LIMIT 10;")
            
            # Add more complex queries - aggregations
            numeric_cols = [c['name'] for c in columns if 
                           str(c['type']).lower().startswith(('int', 'float', 'double', 'decimal', 'numeric'))]
            
            if numeric_cols and len(numeric_cols) >= 1:
                # Aggregation query
                agg_col = numeric_cols[0]
                group_cols = [c['name'] for c in columns if 
                             not str(c['type']).lower().startswith(('int', 'float', 'double', 'decimal', 'numeric'))]
                
                if group_cols:
                    group_col = group_cols[0]
                    if db_type == 'mssql':
                        example_queries.append(
                            f"SELECT [{group_col}], SUM([{agg_col}]) as total_{agg_col}, "
                            f"AVG([{agg_col}]) as avg_{agg_col}, COUNT(*) as count "
                            f"FROM [{table_identifier}] "
                            f"GROUP BY [{group_col}] "
                            f"ORDER BY total_{agg_col} DESC;"
                        )
                    else:
                        example_queries.append(
                            f"SELECT {group_col}, SUM({agg_col}) as total_{agg_col}, "
                            f"AVG({agg_col}) as avg_{agg_col}, COUNT(*) as count "
                            f"FROM {table_identifier} "
                            f"GROUP BY {group_col} "
                            f"ORDER BY total_{agg_col} DESC "
                            f"LIMIT 10;"
                        )
            
            # Try to find relationships for JOIN examples
            try:
                foreign_keys = inspector.get_foreign_keys(table, schema=schema)
                
                for fk in foreign_keys:
                    if 'referred_table' in fk and fk['referred_table']:
                        referred_table = fk['referred_table']
                        referred_schema = fk.get('referred_schema', schema)
                        
                        # Get the referred table's columns
                        referred_columns = inspector.get_columns(referred_table, schema=referred_schema)
                        
                        if not referred_columns:
                            continue
                        
                        # Create a JOIN query
                        refs = fk.get('constrained_columns', [])
                        referred_cols = fk.get('referred_columns', [])
                        
                        if refs and referred_cols:
                            join_condition = []
                            for i in range(min(len(refs), len(referred_cols))):
                                if db_type == 'mssql':
                                    join_condition.append(
                                        f"t1.[{refs[i]}] = t2.[{referred_cols[i]}]"
                                    )
                                else:
                                    join_condition.append(
                                        f"t1.{refs[i]} = t2.{referred_cols[i]}"
                                    )
                            
                            join_cond_str = " AND ".join(join_condition)
                            
                            # Create a qualified table reference
                            ref_table_id = f"{referred_schema}.{referred_table}" if referred_schema else referred_table
                            
                            # Add a few useful columns from both tables
                            t1_cols = [c['name'] for c in columns[:2]]
                            t2_cols = [c['name'] for c in referred_columns[:2]]
                            
                            # Format the column selections
                            if db_type == 'mssql':
                                t1_cols_fmt = ', '.join([f"t1.[{c}]" for c in t1_cols])
                                t2_cols_fmt = ', '.join([f"t2.[{c}]" for c in t2_cols])
                                
                                example_queries.append(
                                    f"SELECT TOP 10 {t1_cols_fmt}, {t2_cols_fmt} "
                                    f"FROM [{table_identifier}] t1 "
                                    f"JOIN [{ref_table_id}] t2 ON {join_cond_str};"
                                )
                            else:
                                t1_cols_fmt = ', '.join([f"t1.{c}" for c in t1_cols])
                                t2_cols_fmt = ', '.join([f"t2.{c}" for c in t2_cols])
                                
                                example_queries.append(
                                    f"SELECT {t1_cols_fmt}, {t2_cols_fmt} "
                                    f"FROM {table_identifier} t1 "
                                    f"JOIN {ref_table_id} t2 ON {join_cond_str} "
                                    f"LIMIT 10;"
                                )
            except Exception as e:
                print(f"Warning: Could not generate JOIN queries for table {table}: {e}")
        
        except Exception as e:
            print(f"Warning: Could not generate example queries for table {table}: {e}")
    
    return example_queries

def save_connection(args, filename="db_connection.json"):
    """Save connection details to a JSON file for reuse"""
    connection_data = {
        "db_type": args.db_type,
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "username": args.username,
        # Don't save password for security reasons
    }
    
    # Add specialized fields based on database type
    if args.db_type == 'bigquery':
        connection_data["project_id"] = args.project_id
        connection_data["credentials_file"] = args.credentials_file
    
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    # Save to file
    with open(os.path.join("config", filename), "w") as f:
        json.dump(connection_data, f, indent=2)
    
    print(f"Connection details saved to config/{filename}")

def load_connection(filename="db_connection.json"):
    """Load connection details from a JSON file"""
    try:
        with open(os.path.join("config", filename), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def generate_business_terminology_file(filename=None):
    """Generate a template for business terminology definitions"""
    if not filename:
        filename = os.path.join(TERMINOLOGY_DIR, "terminology.txt")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("# Business Terminology Definitions\n")
            f.write("# Format: Term: Definition\n\n")
            f.write("# Example:\n")
            f.write("# OTIF: OTIF stands for 'On Time In Full' and measures delivery performance\n\n")
            f.write("# Add your business terms below:\n\n")
        
        print(f"Created business terminology template file at {filename}")
        print("Please edit this file to add your business terminology definitions.")
    else:
        print(f"Business terminology file already exists at {filename}")
    
    return filename

def review_training_examples(vn):
    """Allow user to review and remove training examples"""
    try:
        # Get all training data
        training_data = vn.get_training_data()
        
        if training_data is None or (hasattr(training_data, 'empty') and training_data.empty) or len(training_data) == 0:
            print("No training data found to review.")
            return
        
        # Convert to list if it's a DataFrame
        if hasattr(training_data, 'to_dict'):
            training_data = training_data.to_dict('records')
        
        print(f"\nFound {len(training_data)} training examples:")
        
        # Display in pages of 10
        page_size = 10
        current_page = 0
        total_pages = (len(training_data) + page_size - 1) // page_size
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(training_data))
            
            print(f"\n--- Page {current_page+1} of {total_pages} ---")
            
            for i in range(start_idx, end_idx):
                item = training_data[i]
                item_id = item.get('id', f"item-{i}")
                item_type = item.get('training_data_type', 'unknown')
                
                if isinstance(item_type, str) and item_type.endswith('-sql'):
                    question = item.get('question', 'No question')
                    sql = item.get('content', 'No SQL')
                    print(f"\n{i+1}. [{item_id}] - SQL Example:")
                    print(f"   Q: {question}")
                    print(f"   SQL: {sql[:100]}..." if len(str(sql)) > 100 else f"   SQL: {sql}")
                elif isinstance(item_type, str) and item_type.endswith('-doc'):
                    doc = item.get('content', 'No documentation')
                    print(f"\n{i+1}. [{item_id}] - Documentation:")
                    print(f"   {str(doc)[:100]}..." if len(str(doc)) > 100 else f"   {doc}")
                elif isinstance(item_type, str) and item_type.endswith('-ddl'):
                    ddl = item.get('content', 'No DDL')
                    print(f"\n{i+1}. [{item_id}] - DDL:")
                    print(f"   {str(ddl)[:100]}..." if len(str(ddl)) > 100 else f"   {ddl}")
                else:
                    print(f"\n{i+1}. [{item_id}] - Unknown type: {item_type}")
            
            print("\nCommands:")
            print("n - Next page")
            print("p - Previous page")
            print("d <num> - Delete item by number")
            print("v <num> - View full item")
            print("q - Quit review")
            
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'n' and current_page < total_pages - 1:
                current_page += 1
            elif cmd == 'p' and current_page > 0:
                current_page -= 1
            elif cmd.startswith('d '):
                try:
                    item_num = int(cmd.split(' ')[1]) - 1
                    if 0 <= item_num < len(training_data):
                        item_id = training_data[item_num].get('id')
                        if item_id:
                            confirm = input(f"Are you sure you want to delete item {item_num+1}? (y/n): ")
                            if confirm.lower() == 'y':
                                vn.remove_training_data(id=item_id)
                                print(f"Deleted training example {item_num+1}.")
                                # Refresh the training data after deletion
                                new_training_data = vn.get_training_data()
                                if new_training_data is not None:
                                    # Convert to list if it's a DataFrame
                                    if hasattr(new_training_data, 'to_dict'):
                                        new_training_data = new_training_data.to_dict('records')
                                    training_data = new_training_data
                                    total_pages = (len(training_data) + page_size - 1) // page_size
                                    if current_page >= total_pages and total_pages > 0:
                                        current_page = max(0, total_pages - 1)
                        else:
                            print("Item has no ID and cannot be deleted.")
                    else:
                        print(f"Invalid item number. Please enter a number between 1 and {len(training_data)}.")
                except ValueError:
                    print("Invalid number format. Please enter a valid number.")
            elif cmd.startswith('v '):
                try:
                    item_num = int(cmd.split(' ')[1]) - 1
                    if 0 <= item_num < len(training_data):
                        item = training_data[item_num]
                        print("\nFull item details:")
                        print(json.dumps(item, indent=2, default=str))
                        input("\nPress Enter to continue...")
                    else:
                        print(f"Invalid item number. Please enter a number between 1 and {len(training_data)}.")
                except ValueError:
                    print("Invalid number format. Please enter a valid number.")
    except Exception as e:
        print(f"Error during review: {e}")
        return

def train_vanna(openai_api_key: str, ddl_statements: List[str], example_queries: List[str], 
                engine=None, system_prompt=None, 
                documentation=None, business_terminology=None,
                retrain_from_history=False) -> Any:
    """Train Vanna.AI with extracted schema and example queries"""
    print("Initializing Vanna.AI with OpenAI...")
    
    # Default system prompt if none provided
    default_system_prompt = """You are a SQL expert. Please help to generate a SQL query to answer the question.
Your response should ONLY be based on the given context and follow the response guidelines and format instructions.

===Response Guidelines 
1. If the provided context is sufficient, please generate a valid SQL query for the question.
2. Ensure you generate specific, runnable SQL with actual values wherever possible. Avoid using placeholders like 'sample_value'.
3. If the query requires filtering by a specific value but it's not clear what value to use, generate a query with a reasonable default value based on the data type (e.g., use 1 for integers, 'Example' for strings).
4. If the database has nested data structures, use appropriate extraction functions.
5. Return only the SQL query without explanations.
6. Ensure that the output SQL is SQL-compliant and executable, and free of syntax errors."""

    config = {
        'api_key': openai_api_key,
        'model': 'gpt-4o',  # Use GPT-4o instead of GPT-4-turbo
        'system_prompt': system_prompt if system_prompt else default_system_prompt
    }
    
    print(f"Using model: gpt-4o with {'custom' if system_prompt else 'default'} system prompt")
    
    vn = VannaTrainer(config=config)
    
    # Store the database connection for later use
    if engine:
        vn.maintain_connection(engine)
    
    # Train with DDL statements
    print(f"Training with {len(ddl_statements)} DDL statements...")
    for ddl in ddl_statements:
        try:
            vn.train(ddl=ddl)
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\nError training with DDL: {e}")
    print("\nDDL training complete!")
    
    # Train with example queries
    print(f"Training with {len(example_queries)} example queries...")
    for query in example_queries:
        try:
            vn.train(sql=query)
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\nError training with SQL: {e}")
    print("\nExample query training complete!")
    
    # Add enhanced documentation
    if documentation:
        print(f"Training with {len(documentation)} documentation items...")
        for doc in documentation:
            try:
                vn.train(documentation=doc)
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\nError training with documentation: {e}")
        print("\nDocumentation training complete!")
    else:
        # Add basic documentation about the database
        vn.train(documentation=f"This database contains {len(ddl_statements)} tables with various relationships.")
    
    # Add business terminology
    if business_terminology and len(business_terminology) > 0:
        print(f"Training with {len(business_terminology)} business terminology items...")
        for term in business_terminology:
            try:
                vn.train(documentation=term)
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\nError training with terminology: {e}")
        print("\nBusiness terminology training complete!")
    
    # Retrain from history if requested
    if retrain_from_history:
        print("Retraining from successful query history...")
        vn.retrain_from_history(min_feedback_score=3)  # Only use queries with good feedback
    
    print("Vanna.AI training completed successfully!")
    return vn

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

def create_enhanced_documentation(engine, db_type: str, schema_name=None) -> List[str]:
    """Create enhanced documentation about the database structure"""
    inspector = inspect(engine)
    documentation = []
    
    # Get all schemas if applicable
    if db_type == 'postgresql':
        schemas = inspector.get_schema_names()
        if schema_name:
            schemas = [s for s in schemas if s == schema_name]
    else:
        schemas = [schema_name] if schema_name else [None]
    
    # For each schema
    for schema in schemas:
        if schema in ('pg_catalog', 'information_schema'):
            continue
        
        # Get all tables
        tables = inspector.get_table_names(schema=schema)
        
        for table in tables:
            # Table information
            table_doc = f"Table: {schema + '.' if schema else ''}{table}\n"
            
            # Get columns
            columns = inspector.get_columns(table, schema=schema)
            table_doc += "Columns:\n"
            for col in columns:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                table_doc += f"  - {col['name']}: {col['type']} ({nullable})\n"
            
            # Get primary keys
            try:
                pk = inspector.get_pk_constraint(table, schema=schema)
                if pk and 'constrained_columns' in pk:
                    table_doc += "Primary Key: " + ", ".join(pk['constrained_columns']) + "\n"
            except:
                pass
            
            # Get foreign keys
            try:
                fks = inspector.get_foreign_keys(table, schema=schema)
                if fks:
                    table_doc += "Foreign Keys:\n"
                    for fk in fks:
                        ref_table = fk.get('referred_table', '')
                        ref_schema = fk.get('referred_schema', '')
                        ref_cols = fk.get('referred_columns', [])
                        
                        if ref_table and ref_cols:
                            ref_full = f"{ref_schema + '.' if ref_schema else ''}{ref_table}"
                            cols = ", ".join(fk.get('constrained_columns', []))
                            ref_cols = ", ".join(ref_cols)
                            table_doc += f"  - {cols} -> {ref_full}({ref_cols})\n"
            except:
                pass
            
            # Get indexes
            try:
                indexes = inspector.get_indexes(table, schema=schema)
                if indexes:
                    table_doc += "Indexes:\n"
                    for idx in indexes:
                        idx_name = idx.get('name', 'unnamed')
                        idx_cols = ", ".join(idx.get('column_names', []))
                        unique = "UNIQUE " if idx.get('unique', False) else ""
                        table_doc += f"  - {idx_name}: {unique}({idx_cols})\n"
            except:
                pass
            
            # Add to documentation
            documentation.append(table_doc)
    
    return documentation

def load_business_terminology(terminology_file=None):
    """Load business terminology from a file or prompt user to enter it"""
    terminology = []
    
    # Create directory if it doesn't exist
    os.makedirs(TERMINOLOGY_DIR, exist_ok=True)
    
    if terminology_file and os.path.exists(terminology_file):
        # Load from file
        with open(terminology_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    terminology.append(line)
    else:
        # Default terminology file
        default_file = os.path.join(TERMINOLOGY_DIR, "terminology.txt")
        if os.path.exists(default_file):
            with open(default_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        terminology.append(line)
    
    return terminology

def main():
    parser = argparse.ArgumentParser(description='Automatically train Vanna.AI on any database schema')
    
    # Connection parameters
    connection_group = parser.add_argument_group('Connection Parameters')
    connection_group.add_argument('--connection-string', help='Full database connection string (if provided, overrides other connection parameters)')
    connection_group.add_argument('--connection-file', help='Load connection details from a saved JSON file')
    connection_group.add_argument('--save-connection', help='Save connection details to a file for reuse', action='store_true')
    connection_group.add_argument('--connection-name', help='Name for the saved connection', default='db_connection.json')
    
    # Standard database parameters
    db_group = parser.add_argument_group('Database Parameters')
    db_group.add_argument('--db-type', choices=[
        'postgresql', 'mysql', 'mssql', 'oracle', 'sqlite', 
        'snowflake', 'bigquery', 'redshift', 'duckdb'
    ], help='Database type')
    db_group.add_argument('--host', help='Database host address')
    db_group.add_argument('--port', type=int, help='Database port (default based on db-type)')
    db_group.add_argument('--dbname', help='Database name')
    db_group.add_argument('--username', help='Database username')
    db_group.add_argument('--password', help='Database password (will prompt if not provided)')
    
    # Specialized parameters for specific databases
    special_group = parser.add_argument_group('Specialized Database Parameters')
    special_group.add_argument('--project-id', help='Project ID (for BigQuery)')
    special_group.add_argument('--credentials-file', help='Path to credentials file (for BigQuery, etc.)')
    
    # AI parameters
    ai_group = parser.add_argument_group('AI Parameters')
    ai_group.add_argument('--openai-api-key', help='OpenAI API key (defaults to OPENAI_API_KEY env var)')
    ai_group.add_argument('--openai-model', help='OpenAI model to use', default='gpt-4o')
    ai_group.add_argument('--system-prompt', help='Custom system prompt for the LLM')
    
    # New training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--terminology-file', help='Path to business terminology file')
    training_group.add_argument('--create-terminology-template', help='Create a template for business terminology', action='store_true')
    training_group.add_argument('--enhanced-documentation', help='Generate enhanced documentation for training', action='store_true')
    training_group.add_argument('--retrain-from-history', help='Retrain using successful query history', action='store_true')
    training_group.add_argument('--review-training-data', help='Review and optionally remove training data', action='store_true')
    
    args = parser.parse_args()
    
    # Create business terminology template if requested
    if args.create_terminology_template:
        template_file = generate_business_terminology_file(args.terminology_file)
        print(f"Edit {template_file} with your business terminology and run this script again with --terminology-file={template_file}")
        return 0
    
    # Load saved connection if specified
    if args.connection_file:
        loaded_config = load_connection(args.connection_file)
        if loaded_config:
            # Set args from loaded config (only if not provided in command line)
            for key, value in loaded_config.items():
                if getattr(args, key, None) is None:
                    setattr(args, key, value)
            print(f"Loaded connection details from {args.connection_file}")
        else:
            print(f"Warning: Could not load connection details from {args.connection_file}")
    
    # Validate required parameters
    if not args.connection_string:
        if not args.db_type:
            parser.error("Either --connection-string or --db-type is required")
        
        # Check for required parameters based on database type
        if args.db_type != 'sqlite' and args.db_type != 'duckdb' and not args.host and not args.dbname:
            parser.error(f"For {args.db_type}, --host and --dbname are required")
        
        if args.db_type == 'bigquery' and not args.project_id:
            parser.error("For BigQuery, --project-id is required")
    
    # Set default ports if not provided
    if args.port is None and args.host:
        default_ports = {
            'postgresql': 5432,
            'mysql': 3306,
            'mssql': 1433,
            'oracle': 1521,
            'snowflake': 443,
            'redshift': 5439,
        }
        if args.db_type in default_ports:
            args.port = default_ports[args.db_type]
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        openai_api_key = getpass.getpass("OpenAI API key: ")
    
    # Load business terminology if specified
    business_terminology = None
    if args.terminology_file:
        business_terminology = load_business_terminology(args.terminology_file)
        print(f"Loaded {len(business_terminology)} business terminology items")
    
    print(f"Connecting to {args.db_type} database...")
    conn_string = get_connection_string(args)
    
    # Store connection string in our enhanced storage system
    store_connection_info(
        conn_string, 
        args.db_type, 
        args.host, 
        args.port, 
        args.dbname, 
        args.username
    )
    
    # Save connection details if requested
    if args.save_connection:
        save_connection(args, args.connection_name)
    
    try:
        # Use pooled engine with keep-alive settings
        engine = sqlalchemy.create_engine(
            conn_string, 
            pool_pre_ping=True,           # Check connection before using
            pool_recycle=1800,            # Recycle connections after 30 minutes
            pool_timeout=30,              # Connection timeout after 30 seconds
            pool_size=5,                  # Maintain up to 5 connections
            max_overflow=10               # Allow up to 10 overflow connections
        )
        print("Connection successful!")
        
        # Extract schema
        ddl_statements = extract_schema(engine, args.db_type)
        print(f"Extracted {len(ddl_statements)} DDL statements")
        
        # Extract example queries
        print("Generating example queries...")
        example_queries = extract_example_queries(engine, args.db_type)
        print(f"Generated {len(example_queries)} example queries")
        
        # Generate enhanced documentation if requested
        enhanced_documentation = None
        if args.enhanced_documentation:
            print("Generating enhanced documentation...")
            enhanced_documentation = create_enhanced_documentation(engine, args.db_type)
            print(f"Generated {len(enhanced_documentation)} documentation items")
        
        # Train Vanna.AI and pass the engine to maintain connection
        vn = train_vanna(
            openai_api_key, 
            ddl_statements, 
            example_queries, 
            engine, 
            args.system_prompt,
            documentation=enhanced_documentation,
            business_terminology=business_terminology,
            retrain_from_history=args.retrain_from_history
        )
        
        # Review training data if requested
        if args.review_training_data:
            review_training_examples(vn)
        
        # Example usage
        print("\nTry asking a question about your database:")
        
        # Keep the engine connection for the query session
        session_engine = engine
        
        # Verify connection is active and pass it back to Vanna
        print("Verifying database connection is active for query session...")
        try:
            with session_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✓ Database connection verified")
            # Re-establish connection for vn
            vn.maintain_connection(session_engine)
        except Exception as e:
            print(f"Connection check failed: {e}")
            print("Attempting to reconnect...")
            try:
                session_engine = sqlalchemy.create_engine(
                    conn_string, 
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    pool_timeout=30
                )
                print("New connection established. Setting up with Vanna...")
                vn.maintain_connection(session_engine)
                with session_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print("✓ Reconnection successful")
            except Exception as reconnect_err:
                print(f"Failed to reconnect: {reconnect_err}")
                print("You may experience issues executing queries.")
        
        while True:
            question = input("Ask a question (or 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            
            try:
                # Generate SQL for the question
                sql = vn.ask(question)
                print("\nGenerated SQL:")
                print(sql)
                
                run_query = input("\nRun this query? (y/n): ")
                if run_query.lower() == 'y':
                    try:
                        # Ask for row limit for large result sets
                        limit_input = input("Maximum number of rows to display (default: 10): ")
                        display_limit = int(limit_input) if limit_input.strip() else 10
                        
                        # Check connection and reconnect if needed before executing
                        print("Checking database connection before executing query...")
                        try:
                            # Test connection with a simple query
                            with session_engine.connect() as conn:
                                conn.execute(text("SELECT 1"))
                            print("✓ Connection verified before query execution")
                        except Exception as conn_err:
                            print(f"Connection lost: {conn_err}. Reconnecting...")
                            session_engine = sqlalchemy.create_engine(
                                conn_string, 
                                pool_pre_ping=True,
                                pool_recycle=1800,
                                pool_timeout=30
                            )
                            vn.maintain_connection(session_engine)
                            print("✓ Connection reestablished")
                        
                        # Use the custom execute_query method to ensure connection is maintained
                        print("Executing query...")
                        result = vn.execute_query(sql)
                        
                        if result:
                            rows = result.fetchall()
                            if rows:
                                # Simple display of results
                                columns = result.keys()
                                print("\nResults:")
                                print(' | '.join([str(c) for c in columns]))
                                print('-' * 50)
                                
                                # Show requested number of rows
                                for i, row in enumerate(rows[:display_limit]):
                                    print(' | '.join([str(val) for val in row]))
                                
                                total_rows = len(rows)
                                if total_rows > display_limit:
                                    print(f"\n...and {total_rows - display_limit} more rows")
                                    show_more = input(f"Show all {total_rows} rows? (y/n): ")
                                    if show_more.lower() == 'y':
                                        print("\nAll Results:")
                                        print(' | '.join([str(c) for c in columns]))
                                        print('-' * 50)
                                        for row in rows:
                                            print(' | '.join([str(val) for val in row]))
                                
                                print(f"\nTotal rows: {total_rows}")
                            else:
                                print("Query returned no results")
                        else:
                            print("Error: No result returned from query")
                        
                        # Ask for feedback on the query to save it for retraining
                        save_query = input("\nSave this question-SQL pair for improving future results? (y/n): ")
                        if save_query.lower() == 'y':
                            rating = input("Rate the quality of this SQL (1-5, where 5 is excellent): ")
                            try:
                                rating_value = int(rating)
                                if 1 <= rating_value <= 5:
                                    vn.save_successful_query(question, sql, rating_value)
                                    print("Query saved for future training.")
                                else:
                                    print("Invalid rating. Must be between 1-5.")
                            except ValueError:
                                print("Invalid rating format. Using default score of 3.")
                                vn.save_successful_query(question, sql, 3)
                    except Exception as e:
                        print(f"Error executing query: {e}")
                        print("Reconnecting to database...")
                        try:
                            session_engine = sqlalchemy.create_engine(
                                conn_string, 
                                pool_pre_ping=True,
                                pool_recycle=1800,
                                pool_timeout=30
                            )
                            vn.maintain_connection(session_engine)
                            print("Reconnected successfully.")
                        except Exception as reconnect_err:
                            print(f"Failed to reconnect: {reconnect_err}")
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "-" * 50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 