#!/bin/bash
# Vanna.AI - Unified Script
# This script provides a single interface for training and querying Vanna.AI models

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${GREEN}"
echo "  _    __                        ___    ____"
echo " | |  / /___ _____  ____  ____ _/   |  /  _/"
echo " | | / / __ \`/ __ \/ __ \/ __ \`/ /| |  / /  "
echo " | |/ / /_/ / / / / / / / /_/ / ___ |_/ /   "
echo " |___/\__,_/_/ /_/_/ /_/\__,_/_/  |_/___/   "
echo "                                           "
echo -e "${NC}Unified Training & Query Interface"
echo -e "----------------------------------------${NC}\n"

# Set up directories
mkdir -p config query_history business_terminology

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv || { echo -e "${RED}Failed to create virtual environment. Please install venv module.${NC}"; exit 1; }
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
    touch venv/.requirements_installed
fi

# Check for OPENAI_API_KEY environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}OPENAI_API_KEY environment variable not set.${NC}"
    read -sp "Please enter your OpenAI API key: " OPENAI_API_KEY
    echo ""
    export OPENAI_API_KEY=$OPENAI_API_KEY
fi

# Parse command line arguments
MODE="interactive"
DB_TYPE=""
DB_HOST=""
DB_PORT=""
DB_NAME=""
DB_USER=""
SKIP_TERMINOLOGY=false
SKIP_DOCUMENTATION=false
SKIP_REVIEW=false
SKIP_RETRAIN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --train)
      MODE="train"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --db-type)
      DB_TYPE="$2"
      shift 2
      ;;
    --host)
      DB_HOST="$2"
      shift 2
      ;;
    --port)
      DB_PORT="$2"
      shift 2
      ;;
    --dbname)
      DB_NAME="$2"
      shift 2
      ;;
    --username)
      DB_USER="$2"
      shift 2
      ;;
    --skip-terminology)
      SKIP_TERMINOLOGY=true
      shift
      ;;
    --skip-documentation)
      SKIP_DOCUMENTATION=true
      shift
      ;;
    --skip-review)
      SKIP_REVIEW=true
      shift
      ;;
    --skip-retrain)
      SKIP_RETRAIN=true
      shift
      ;;
    --help)
      echo -e "${BLUE}Vanna.AI Unified Interface${NC}"
      echo "Usage: $0 [options]"
      echo ""
      echo "Mode options:"
      echo "  --train                  Train a new model or enhance existing model"
      echo "  --run                    Run queries against a previously trained model"
      echo "  (No mode flag will start interactive mode)"
      echo ""
      echo "Database connection options:"
      echo "  --db-type TYPE           Database type (postgresql, mysql, etc.)"
      echo "  --host HOST              Database hostname"
      echo "  --port PORT              Database port"
      echo "  --dbname NAME            Database name"
      echo "  --username USER          Database username"
      echo ""
      echo "Enhanced training options:"
      echo "  --skip-terminology       Skip business terminology setup"
      echo "  --skip-documentation     Skip enhanced documentation generation"
      echo "  --skip-review            Skip training data review"
      echo "  --skip-retrain           Skip retraining from history"
      echo ""
      echo "Examples:"
      echo "  $0                       Start in interactive mode"
      echo "  $0 --train --db-type postgresql --host localhost --dbname mydb --username user"
      echo "  $0 --run                 Run queries using previously trained model"
      echo ""
      deactivate
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Run $0 --help for usage information."
      deactivate
      exit 1
      ;;
  esac
done

# Interactive mode if no mode specified
if [ "$MODE" = "interactive" ]; then
    echo -e "${BLUE}Welcome to Vanna.AI Interactive Mode${NC}"
    echo -e "Please select an operation:"
    echo -e "  ${GREEN}1${NC}. Train a new model / Enhance existing model"
    echo -e "  ${GREEN}2${NC}. Run queries against a trained model"
    echo -e "  ${GREEN}0${NC}. Exit"
    
    read -p "Enter your choice (0-2): " CHOICE
    
    case $CHOICE in
        1)
            MODE="train"
            ;;
        2)
            MODE="run"
            ;;
        0)
            echo "Exiting. Goodbye!"
            deactivate
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            deactivate
            exit 1
            ;;
    esac
fi

# Function to handle training mode
handle_training_mode() {
    echo -e "\n${BLUE}=== Vanna.AI Training Mode ===${NC}"
    
    # Connection setup
    echo -e "\n${BLUE}=== Database Connection Setup ===${NC}"
    
    # Check if we have a saved connection
    if [ -f "config/db_connection.json" ] && [ -z "$DB_TYPE" ] && [ -z "$DB_HOST" ]; then
        echo -e "${GREEN}✓ Found saved database connection${NC}"
        CONNECTION_ARG="--connection-file=db_connection.json"
    else
        # Validate required parameters for new connections
        if [ -z "$DB_TYPE" ] || [ -z "$DB_HOST" ] || [ -z "$DB_NAME" ] || [ -z "$DB_USER" ]; then
            # If command-line args weren't provided, ask interactively
            if [ -z "$DB_TYPE" ]; then
                echo "Select database type:"
                echo "1. PostgreSQL"
                echo "2. MySQL"
                echo "3. Microsoft SQL Server"
                echo "4. Oracle"
                echo "5. SQLite"
                echo "6. Snowflake"
                echo "7. BigQuery"
                read -p "Enter choice (1-7): " DB_TYPE_CHOICE
                
                case $DB_TYPE_CHOICE in
                    1) DB_TYPE="postgresql" ;;
                    2) DB_TYPE="mysql" ;;
                    3) DB_TYPE="mssql" ;;
                    4) DB_TYPE="oracle" ;;
                    5) DB_TYPE="sqlite" ;;
                    6) DB_TYPE="snowflake" ;;
                    7) DB_TYPE="bigquery" ;;
                    *) echo -e "${RED}Invalid choice${NC}"; deactivate; exit 1 ;;
                esac
            fi
            
            if [ "$DB_TYPE" != "sqlite" ] && [ -z "$DB_HOST" ]; then
                read -p "Enter database host: " DB_HOST
            fi
            
            if [ "$DB_TYPE" != "sqlite" ] && [ -z "$DB_PORT" ]; then
                # Default ports
                case $DB_TYPE in
                    postgresql) DEFAULT_PORT=5432 ;;
                    mysql) DEFAULT_PORT=3306 ;;
                    mssql) DEFAULT_PORT=1433 ;;
                    oracle) DEFAULT_PORT=1521 ;;
                    snowflake) DEFAULT_PORT=443 ;;
                    redshift) DEFAULT_PORT=5439 ;;
                    *) DEFAULT_PORT="" ;;
                esac
                
                if [ ! -z "$DEFAULT_PORT" ]; then
                    read -p "Enter database port (default: $DEFAULT_PORT): " DB_PORT
                    if [ -z "$DB_PORT" ]; then
                        DB_PORT=$DEFAULT_PORT
                    fi
                fi
            fi
            
            if [ -z "$DB_NAME" ]; then
                read -p "Enter database name: " DB_NAME
            fi
            
            if [ "$DB_TYPE" != "sqlite" ] && [ -z "$DB_USER" ]; then
                read -p "Enter database username: " DB_USER
            fi
        fi
        
        echo -e "${YELLOW}Creating new database connection...${NC}"
        
        # Build connection arguments
        CONNECTION_ARG="--db-type=$DB_TYPE"
        
        if [ "$DB_TYPE" != "sqlite" ]; then
            CONNECTION_ARG="$CONNECTION_ARG --host=$DB_HOST --dbname=$DB_NAME --username=$DB_USER"
            if [ ! -z "$DB_PORT" ]; then
                CONNECTION_ARG="$CONNECTION_ARG --port=$DB_PORT"
            fi
        else
            CONNECTION_ARG="$CONNECTION_ARG --dbname=$DB_NAME"
        fi
        
        # Add save connection flag
        CONNECTION_ARG="$CONNECTION_ARG --save-connection"
    fi
    
    # Step 1: Business Terminology Setup
    if [ "$SKIP_TERMINOLOGY" = false ]; then
        echo -e "\n${BLUE}=== Step 1: Business Terminology Setup ===${NC}"
        
        if [ ! -f "business_terminology/terminology.txt" ]; then
            echo -e "${YELLOW}Creating business terminology template...${NC}"
            python3 train_vanna.py --create-terminology-template
            
            echo -e "\n${YELLOW}Please edit business_terminology/terminology.txt with your business terms${NC}"
            echo -e "Format: Term: Definition"
            echo -e "Example: OTIF: On Time In Full delivery performance metric"
            
            read -p "Press Enter when you've edited the terminology file (or Ctrl+C to exit)... "
        else
            echo -e "${GREEN}✓ Business terminology file exists${NC}"
            echo -e "Location: business_terminology/terminology.txt"
            
            # Show preview of terminology
            echo -e "\n${YELLOW}Preview of terminology:${NC}"
            head -n 5 business_terminology/terminology.txt
            echo "..."
        fi
        
        TERMINOLOGY_ARG="--terminology-file=business_terminology/terminology.txt"
    else
        TERMINOLOGY_ARG=""
        echo -e "\n${YELLOW}Skipping business terminology setup${NC}"
    fi
    
    # Step A command base
    BASE_CMD="python3 train_vanna.py $CONNECTION_ARG"
    
    # Step 2: Enhanced Documentation
    if [ "$SKIP_DOCUMENTATION" = false ]; then
        echo -e "\n${BLUE}=== Step 2: Enhanced Documentation Generation ===${NC}"
        echo -e "${YELLOW}Generating comprehensive documentation of database structure...${NC}"
        DOCUMENTATION_ARG="--enhanced-documentation"
    else
        DOCUMENTATION_ARG=""
        echo -e "\n${YELLOW}Skipping enhanced documentation generation${NC}"
    fi
    
    # Step 3: Review Training Data
    REVIEW_ARG=""
    if [ "$SKIP_REVIEW" = false ]; then
        echo -e "\n${BLUE}=== Step 3: Training Data Quality Review ===${NC}"
        echo -e "${YELLOW}Setting up review of existing training data${NC}"
        REVIEW_ARG="--review-training-data"
    else
        echo -e "\n${YELLOW}Skipping training data review${NC}"
    fi
    
    # Step 4: Retrain from History
    RETRAIN_ARG=""
    if [ "$SKIP_RETRAIN" = false ]; then
        echo -e "\n${BLUE}=== Step 4: Retraining from Successful Queries ===${NC}"
        
        if [ -d "query_history" ] && [ "$(ls -A query_history 2>/dev/null)" ]; then
            echo -e "${GREEN}✓ Found saved query history${NC}"
            RETRAIN_ARG="--retrain-from-history"
        else
            echo -e "${YELLOW}No query history found to retrain from${NC}"
        fi
    else
        echo -e "\n${YELLOW}Skipping retraining from history${NC}"
    fi
    
    # Build complete command
    FULL_CMD="$BASE_CMD $TERMINOLOGY_ARG $DOCUMENTATION_ARG $REVIEW_ARG $RETRAIN_ARG"
    
    # Summary
    echo -e "\n${BLUE}=== Ready to Proceed ===${NC}"
    echo -e "The following enhancements will be applied:"
    [ "$SKIP_TERMINOLOGY" = false ] && echo -e "${GREEN}✓ Business terminology integration${NC}" || echo -e "${YELLOW}✗ Business terminology integration (skipped)${NC}"
    [ "$SKIP_DOCUMENTATION" = false ] && echo -e "${GREEN}✓ Enhanced documentation generation${NC}" || echo -e "${YELLOW}✗ Enhanced documentation generation (skipped)${NC}"
    [ "$SKIP_REVIEW" = false ] && echo -e "${GREEN}✓ Training data quality review${NC}" || echo -e "${YELLOW}✗ Training data quality review (skipped)${NC}"
    [ "$SKIP_RETRAIN" = false -a -n "$RETRAIN_ARG" ] && echo -e "${GREEN}✓ Retraining from query history${NC}" || echo -e "${YELLOW}✗ Retraining from query history (skipped or no history)${NC}"
    
    echo -e "\n${YELLOW}Command:${NC} $FULL_CMD"
    
    # Save the command for future reference
    echo "$FULL_CMD" > config/last_command.txt
    
    # Confirmation
    read -p "Proceed with Vanna.AI training? (y/n): " confirm
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        echo -e "${RED}Training cancelled${NC}"
        deactivate
        exit 0
    fi
    
    # Run the command
    echo -e "\n${BLUE}=== Starting Vanna.AI Training ===${NC}"
    echo -e "${YELLOW}This process will optimize Vanna.AI for your database and use cases${NC}"
    echo -e "${YELLOW}Starting training process...${NC}\n"
    
    eval $FULL_CMD
    
    echo -e "\n${GREEN}=== Vanna.AI Training Complete ===${NC}"
    echo -e "Your Vanna.AI implementation now includes:"
    echo -e "• Business terminology understanding"
    echo -e "• Enhanced documentation of your database structure"
    echo -e "• Quality-controlled training examples"
    echo -e "• Learning from past successful queries"
    echo -e "\nAs you use Vanna.AI, don't forget to rate successful queries to"
    echo -e "continuously improve the system's performance."
    echo -e "\nTo use your trained model, run: ${BLUE}./vanna_ai.sh --run${NC}"
}

# Function to handle run mode
handle_run_mode() {
    echo -e "\n${BLUE}=== Vanna.AI Query Mode ===${NC}"
    
    # Check if last command file exists
    if [ ! -f "config/last_command.txt" ]; then
        echo -e "${RED}No previous training command found. Please train a model first.${NC}"
        echo -e "You can train a model with: ${BLUE}./vanna_ai.sh --train${NC}"
        deactivate
        exit 1
    fi
    
    # Get the last command
    LAST_COMMAND=$(cat config/last_command.txt)
    
    echo -e "${YELLOW}Reconnecting to database with previous settings...${NC}"
    echo -e "Using command: ${BLUE}$LAST_COMMAND${NC}\n"
    
    # Run the same command
    eval "$LAST_COMMAND"
}

# Execute the selected mode
if [ "$MODE" = "train" ]; then
    handle_training_mode
elif [ "$MODE" = "run" ]; then
    handle_run_mode
else
    echo -e "${RED}Invalid mode. This should not happen.${NC}"
    deactivate
    exit 1
fi

# Deactivate virtual environment
deactivate

echo -e "\n${GREEN}Vanna.AI session complete! Goodbye!${NC}" 