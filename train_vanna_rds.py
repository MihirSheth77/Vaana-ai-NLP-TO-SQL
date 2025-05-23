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
from sqlalchemy import inspect, text
from typing import Dict, List, Optional, Tuple, Any
import glob

# Import Vanna modules
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class VannaTrainer(ChromaDB_VectorStore, OpenAI_Chat):
    """Custom Vanna implementation using ChromaDB and OpenAI"""
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Custom prompt template for Vanna.AI that injects relevant Q&A pairs, DDL, documentation, and the user question.
        """
        # Build examples section
        examples = ""
        for example in question_sql_list:
            if example and "question" in example and "sql" in example:
                examples += f"Q: {example['question']}\nA: {example['sql']}\n\n"
        if not examples:
            examples = "No previous Q&A pairs available.\n"

        # Build DDL section
        ddl = "\n".join(ddl_list) if ddl_list else "No DDL available."
        # Build documentation section
        docs = "\n".join(doc_list) if doc_list else "No documentation available."

        # Use a clear, modular prompt template
        prompt_template = f"""
You are a SQL expert. Use the following context to answer the user's question with a SQL query.

=== Previous Q&A ===
{examples}
=== DDL ===
{ddl}
=== Documentation ===
{docs}
=== User Question ===
{question}

=== Instructions ===
- Only return SQL, no explanations.
- Use the most relevant tables and columns.
- If you can't answer, say why.
- If the question has been asked and answered before, repeat the answer exactly as before.
- Ensure the output SQL is valid and executable.
"""
        # Return as a message log for OpenAI Chat
        return [self.system_message(prompt_template)]

def get_connection_string(args) -> str:
    """Build a connection string from provided arguments"""
    password = args.password or getpass.getpass("Database password: ")
    
    if args.connection_string:
        # User provided full connection string
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
        # For BigQuery, we'll use the credentials file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.credentials_file
        return f"bigquery://{args.project_id}/{args.dbname}"
    elif args.db_type == 'redshift':
        return f"redshift+psycopg2://{args.username}:{password}@{args.host}:{args.port}/{args.dbname}"
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
    """Extract example queries based on database structure"""
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
            
            # WHERE clause for a column
            if columns[0]['name']:
                col = columns[0]['name']
                if db_type == 'mssql':
                    example_queries.append(f"SELECT TOP 10 * FROM [{table_identifier}] WHERE [{col}] = 'sample_value';")
                else:
                    example_queries.append(f"SELECT * FROM {table_identifier} WHERE {col} = 'sample_value' LIMIT 10;")
        
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

def load_contextual_documents(directory="context_docs"):
    """Load all .txt and .md files from the given directory as context docs"""
    docs = []
    if not os.path.exists(directory):
        print(f"No contextual documents directory found at '{directory}'. Skipping context injection.")
        return docs
    for filepath in glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def train_vanna(openai_api_key: str, ddl_statements: List[str], example_queries: List[str]) -> None:
    """Train Vanna.AI with extracted schema and example queries, plus contextual documents"""
    print("Initializing Vanna.AI...")
    vn = VannaTrainer(config={
        'api_key': openai_api_key,
        'model': 'gpt-4-turbo-preview'  # Or another appropriate model
    })

    # Inject contextual documents before DDL and example queries
    contextual_docs = load_contextual_documents()
    if contextual_docs:
        print(f"Injecting {len(contextual_docs)} contextual documents into Vanna.AI...")
        for doc in contextual_docs:
            try:
                vn.train(documentation=doc)
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\nError injecting context doc: {e}")
        print("\nContextual document injection complete!")
    else:
        print("No contextual documents found to inject.")

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
    
    # Add documentation about the database
    vn.train(documentation=f"This database contains {len(ddl_statements)} tables with various relationships.")
    
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
    
    # OpenAI parameters
    ai_group = parser.add_argument_group('AI Parameters')
    ai_group.add_argument('--openai-api-key', help='OpenAI API key (defaults to OPENAI_API_KEY env var)')
    ai_group.add_argument('--openai-model', help='OpenAI model to use', default='gpt-4-turbo-preview')
    
    args = parser.parse_args()
    
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
    
    print(f"Connecting to {args.db_type} database...")
    conn_string = get_connection_string(args)
    
    # Save connection details if requested
    if args.save_connection:
        save_connection(args, args.connection_name)
    
    try:
        engine = sqlalchemy.create_engine(conn_string)
        print("Connection successful!")
        
        # Extract schema
        ddl_statements = extract_schema(engine, args.db_type)
        print(f"Extracted {len(ddl_statements)} DDL statements")
        
        # Extract example queries
        print("Generating example queries...")
        example_queries = extract_example_queries(engine, args.db_type)
        print(f"Generated {len(example_queries)} example queries")
        
        # Train Vanna.AI
        vn = train_vanna(openai_api_key, ddl_statements, example_queries)
        
        # Example usage
        print("\nTry asking a question about your database:")
        while True:
            question = input("Ask a question (or 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            
            try:
                sql = vn.ask(question)
                print("\nGenerated SQL:")
                print(sql)
                
                run_query = input("\nRun this query? (y/n): ")
                if run_query.lower() == 'y':
                    with engine.connect() as conn:
                        result = conn.execute(text(sql))
                        rows = result.fetchall()
                        if rows:
                            # Simple display of results
                            columns = result.keys()
                            print("\nResults:")
                            print(' | '.join([str(c) for c in columns]))
                            print('-' * 50)
                            for row in rows[:10]:  # Show first 10 rows
                                print(' | '.join([str(val) for val in row]))
                            
                            if len(rows) > 10:
                                print(f"...and {len(rows) - 10} more rows")
                        else:
                            print("Query returned no results")
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "-" * 50 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 