# Vanna.AI Database Automation

This tool automatically extracts the schema from any supported database and trains a Vanna.AI model to generate SQL from natural language questions.

## Features

- **Universal Database Support**: Connect to almost any SQL database (PostgreSQL, MySQL, SQL Server, Oracle, Snowflake, BigQuery, etc.)
- **Automatic Schema Extraction**: Extracts table definitions, columns, and relationships
- **Example Query Generation**: Creates example queries for better training
- **Connection Saving**: Save connection details for easy reuse
- **Interactive Query Interface**: Ask questions and see the SQL + results

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The simplest way to use this tool is with the provided shell scripts:

```bash
# For training a new model
./train_vanna.sh --db-type postgresql --host your-db-host.com --dbname your_database --username your_username

# For running queries with a previously trained model
./run_vanna.sh
```

### Advanced Command-Line Options

```bash
python train_vanna.py --db-type postgresql --host your-db-host.com --dbname your_database --username your_username
```

## Supported Database Types

- `postgresql` - PostgreSQL 
- `mysql` - MySQL databases
- `mssql` - Microsoft SQL Server
- `oracle` - Oracle Database
- `sqlite` - SQLite
- `snowflake` - Snowflake
- `bigquery` - Google BigQuery
- `redshift` - Amazon Redshift
- `duckdb` - DuckDB

## Parameters

### Basic Connection Parameters
- `--connection-string`: Full database connection string (if provided, overrides other connection parameters)
- `--connection-file`: Load connection details from a saved JSON file
- `--save-connection`: Save connection details to a file for reuse
- `--connection-name`: Name for the saved connection (default: db_connection.json)

### Standard Database Parameters
- `--db-type`: Database type (required if not using connection string or file)
- `--host`: Database host address
- `--port`: Database port (optional, defaults based on db-type)
- `--dbname`: Database name
- `--username`: Database username
- `--password`: Database password (if not provided, will prompt)

### Specialized Database Parameters
- `--project-id`: Project ID (for BigQuery)
- `--credentials-file`: Path to credentials file (for BigQuery, etc.)

### AI Parameters
- `--openai-api-key`: OpenAI API key (defaults to OPENAI_API_KEY env var)
- `--openai-model`: OpenAI model to use (default: gpt-4-turbo-preview)

## Example Commands

```bash
# PostgreSQL
./train_vanna.sh --db-type postgresql --host pg-db.example.com --dbname customers --username admin

# MySQL
./train_vanna.sh --db-type mysql --host mysql-db.example.com --dbname orders --username admin --port 3306

# SQLite
./train_vanna.sh --db-type sqlite --dbname ./my_local_database.db

# Snowflake
./train_vanna.sh --db-type snowflake --host myorg-account.snowflakecomputing.com --dbname ANALYTICS --username snowuser

# BigQuery
./train_vanna.sh --db-type bigquery --project-id my-gcp-project --dbname mydataset --credentials-file ./key.json

# Using a saved connection
./train_vanna.sh --connection-file my_saved_connection.json
```

## Workflow

After training, you can ask questions like:

- "Show me the top 10 customers by revenue"
- "What was the total sales in the last month?"
- "How many orders were placed last week?"

The script will generate SQL and give you the option to execute it directly.

## Notes

- The extracted schema information is used for training the model locally
- Your database contents are never sent to the LLM
- The script supports automatic generation of example queries based on your schema
- Vector database storage is handled through ChromaDB by default 