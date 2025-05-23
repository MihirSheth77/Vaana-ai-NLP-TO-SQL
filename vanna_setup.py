#!/usr/bin/env python3
"""
Vanna AI Setup Wizard
---------------------
Interactive setup wizard that automates the entire Vanna.AI training and query process.
Just provide your database credentials once, and everything else is automated.
"""

import os
import sys
import json
import getpass
import argparse
import subprocess
from typing import Dict, Optional, List, Tuple, Any

# Terminal colors for better UX
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_banner():
    """Print the welcome banner"""
    print(f"""
{Colors.BLUE}{Colors.BOLD}==============================================={Colors.END}
{Colors.BLUE}{Colors.BOLD}        VANNA.AI DATABASE AUTOMATION          {Colors.END}
    
This wizard will help you:
1. Connect to your database
2. Extract the schema automatically
3. Train a Vanna.AI model on your data
4. Let you ask questions in natural language
    """)

def check_prerequisites():
    """Check if all prerequisites are installed"""
    # Check if Python is installed correctly
    print(f"{Colors.BOLD}Checking prerequisites...{Colors.END}")
    
    # Check for virtual environment
    if not os.path.exists("venv"):
        print(f"{Colors.YELLOW}Creating Python virtual environment...{Colors.END}")
        try:
            subprocess.run(["python3", "-m", "venv", "venv"], check=True)
        except subprocess.CalledProcessError:
            print(f"{Colors.RED}Failed to create virtual environment. Please install venv module.{Colors.END}")
            sys.exit(1)
    
    # Install requirements
    print(f"{Colors.YELLOW}Installing requirements...{Colors.END}")
    subprocess.run([
        f"{'venv/Scripts/pip' if sys.platform == 'win32' else 'venv/bin/pip'}", 
        "install", "-r", "requirements.txt"
    ])
    
    # Create necessary directories
    os.makedirs("config", exist_ok=True)
    
    print(f"{Colors.GREEN}All prerequisites satisfied!{Colors.END}")

def get_db_type() -> str:
    """Ask user for database type"""
    db_types = {
        "1": "postgresql",
        "2": "mysql",
        "3": "mssql",
        "4": "sqlite",
        "5": "oracle",
        "6": "snowflake",
        "7": "bigquery",
        "8": "redshift",
        "9": "duckdb",
        "0": "other"
    }
    
    print(f"\n{Colors.BOLD}What type of database are you using?{Colors.END}")
    print("1. PostgreSQL")
    print("2. MySQL")
    print("3. Microsoft SQL Server")
    print("4. SQLite")
    print("5. Oracle")
    print("6. Snowflake")
    print("7. Google BigQuery")
    print("8. Amazon Redshift")
    print("9. DuckDB")
    print("0. Other (direct connection string)")
    
    while True:
        choice = input(f"\n{Colors.BOLD}Enter your choice (1-9, 0): {Colors.END}")
        if choice in db_types:
            return db_types[choice]
        print(f"{Colors.RED}Invalid choice. Please try again.{Colors.END}")

def get_connection_details(db_type: str) -> Dict[str, Any]:
    """Get database connection details based on database type"""
    connection = {"db_type": db_type}
    
    if db_type == "other":
        connection["connection_string"] = input(f"{Colors.BOLD}Enter your full connection string: {Colors.END}")
        return connection
    
    if db_type == "sqlite":
        connection["dbname"] = input(f"{Colors.BOLD}Enter path to SQLite database file: {Colors.END}")
        return connection
    
    if db_type == "duckdb":
        connection["dbname"] = input(f"{Colors.BOLD}Enter path to DuckDB database file: {Colors.END}")
        return connection
    
    if db_type == "bigquery":
        connection["project_id"] = input(f"{Colors.BOLD}Enter your GCP Project ID: {Colors.END}")
        connection["dbname"] = input(f"{Colors.BOLD}Enter dataset name: {Colors.END}")
        credentials_path = input(f"{Colors.BOLD}Enter path to credentials JSON file: {Colors.END}")
        connection["credentials_file"] = os.path.abspath(credentials_path)
        return connection
    
    # For most database types, we need host, port, dbname, username, password
    connection["host"] = input(f"{Colors.BOLD}Enter database host: {Colors.END}")
    
    # Set default port based on db type
    default_ports = {
        "postgresql": "5432",
        "mysql": "3306",
        "mssql": "1433",
        "oracle": "1521",
        "snowflake": "443",
        "redshift": "5439"
    }
    
    default_port = default_ports.get(db_type, "")
    port_input = input(f"{Colors.BOLD}Enter port [{default_port}]: {Colors.END}")
    connection["port"] = port_input if port_input else default_port
    
    connection["dbname"] = input(f"{Colors.BOLD}Enter database name: {Colors.END}")
    connection["username"] = input(f"{Colors.BOLD}Enter username: {Colors.END}")
    
    # Don't save password in settings
    connection["password"] = getpass.getpass(f"{Colors.BOLD}Enter password (will not be stored): {Colors.END}")
    
    # Special case for Snowflake
    if db_type == "snowflake":
        connection["account"] = input(f"{Colors.BOLD}Enter Snowflake account name: {Colors.END}")
    
    return connection

def save_connection(connection: Dict[str, Any], filename: str = "default_connection.json"):
    """Save connection details to a file, excluding password"""
    # Create a copy without password
    to_save = {k: v for k, v in connection.items() if k != "password"}
    
    with open(os.path.join("config", filename), "w") as f:
        json.dump(to_save, f, indent=2)
    
    print(f"{Colors.GREEN}Connection details saved to config/{filename}{Colors.END}")

def build_command(connection: Dict[str, Any]) -> List[str]:
    """Build the command to run the trainer script"""
    # Start with the Python executable from the virtual environment
    python_exe = "venv/Scripts/python" if sys.platform == "win32" else "venv/bin/python"
    
    cmd = [python_exe, "train_vanna.py"]
    
    # Add connection string if provided
    if "connection_string" in connection:
        cmd.extend(["--connection-string", connection["connection_string"]])
        return cmd
    
    # Add database type
    cmd.extend(["--db-type", connection["db_type"]])
    
    # Add standard parameters if present
    if "host" in connection:
        cmd.extend(["--host", connection["host"]])
    
    if "port" in connection:
        cmd.extend(["--port", connection["port"]])
    
    if "dbname" in connection:
        cmd.extend(["--dbname", connection["dbname"]])
    
    if "username" in connection:
        cmd.extend(["--username", connection["username"]])
    
    # Add password if provided
    if "password" in connection:
        cmd.extend(["--password", connection["password"]])
    
    # Special parameters for BigQuery
    if connection["db_type"] == "bigquery":
        if "project_id" in connection:
            cmd.extend(["--project-id", connection["project_id"]])
        if "credentials_file" in connection:
            cmd.extend(["--credentials-file", connection["credentials_file"]])
    
    # Save the connection for future use
    cmd.extend(["--save-connection", "--connection-name", "default_connection.json"])
    
    # Configure OpenAI model with proper system prompt
    openai_prompt = """You are a SQL expert responsible for generating accurate SQL queries from natural language questions.

The user will provide you with questions about their database. The context includes database schema information (table names, column names, relationships) that you should use to craft your SQL queries.

Instructions:
1. Focus exclusively on generating valid SQL queries based on the database schema in context
2. Return ONLY the SQL query with no explanations or markdown
3. Use table aliases to improve readability when joining multiple tables
4. Format your SQL with proper capitalization of SQL keywords and include line breaks for readability
5. Ensure all column references are qualified with table names or aliases when multiple tables are involved
6. Write queries that will execute successfully in PostgreSQL syntax
"""
    
    cmd.extend(["--system-prompt", openai_prompt])
    
    # Get OpenAI API key if not in environment
    if "OPENAI_API_KEY" not in os.environ:
        api_key = getpass.getpass(f"{Colors.BOLD}Enter your OpenAI API key: {Colors.END}")
        if api_key:
            cmd.extend(["--openai-api-key", api_key])
            # Save temporarily for this session
            os.environ["OPENAI_API_KEY"] = api_key
    
    return cmd

def run_training(cmd: List[str]):
    """Run the training command"""
    print(f"\n{Colors.BOLD}Starting Vanna.AI training with Claude 3.7 Sonnet...{Colors.END}")
    
    # Save the command for later (without password)
    safe_cmd = []
    skip_next = False
    for i, arg in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if arg == "--password" or arg == "--openai-api-key":
            skip_next = True
            continue
        safe_cmd.append(arg)
    
    # Save safe command to file
    with open(os.path.join("config", "last_command.txt"), "w") as f:
        f.write(" ".join(safe_cmd))
    
    # Run the command
    try:
        subprocess.run(cmd)
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error during training: {e}{Colors.END}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Training interrupted by user.{Colors.END}")
        sys.exit(1)

def run_query_mode():
    """Run in query mode using previously saved settings"""
    print(f"\n{Colors.BOLD}Starting Vanna.AI query mode with Claude 3.7 Sonnet...{Colors.END}")
    
    # Check if last command exists
    last_cmd_path = os.path.join("config", "last_command.txt")
    if not os.path.exists(last_cmd_path):
        print(f"{Colors.RED}No previous training found. Please run training first.{Colors.END}")
        sys.exit(1)
    
    # Read the command
    with open(last_cmd_path, "r") as f:
        last_cmd = f.read().strip()
    
    # Run the command
    try:
        subprocess.run(last_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error during query: {e}{Colors.END}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Query mode interrupted by user.{Colors.END}")
        sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Vanna.AI Setup Wizard")
    parser.add_argument("--query", action="store_true", 
                        help="Run in query mode using previously saved settings")
    parser.add_argument("--autorun", action="store_true",
                        help="Automatically run the previously saved configuration")
    
    args = parser.parse_args()
    
    # Handle query mode
    if args.query:
        check_prerequisites()
        run_query_mode()
        return
    
    # Handle autorun mode
    if args.autorun:
        check_prerequisites()
        connection_path = os.path.join("config", "default_connection.json")
        
        if not os.path.exists(connection_path):
            print(f"{Colors.RED}No saved connection found. Please run setup first.{Colors.END}")
            sys.exit(1)
        
        with open(connection_path, "r") as f:
            connection = json.load(f)
            
        # Ask for password
        connection["password"] = getpass.getpass(f"{Colors.BOLD}Enter database password: {Colors.END}")
        
        # Build and run command
        cmd = build_command(connection)
        run_training(cmd)
        return
    
    # Regular setup mode
    print_banner()
    check_prerequisites()
    
    # Get database connection details
    db_type = get_db_type()
    connection = get_connection_details(db_type)
    
    # Save connection details
    save_connection(connection)
    
    # Build and run command
    cmd = build_command(connection)
    run_training(cmd)
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}Setup complete!{Colors.END}")
    print(f"\nTo run queries in the future, simply use: {Colors.BOLD}python vanna_setup.py --query{Colors.END}")
    print(f"To run with the same database (only prompting for password): {Colors.BOLD}python vanna_setup.py --autorun{Colors.END}")

if __name__ == "__main__":
    main() 