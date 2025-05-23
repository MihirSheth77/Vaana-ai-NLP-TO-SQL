#!/bin/bash

# Vanna.AI Query Script
# This script runs queries against a previously trained Vanna.AI model

# Check if config directory exists
if [ ! -d "config" ]; then
    echo "Config directory not found. Please run training first with ./train_vanna.sh"
    exit 1
fi

# Check if last command file exists
if [ ! -f "config/last_command.txt" ]; then
    echo "No previous training command found. Please run training first with ./train_vanna.sh"
    exit 1
fi

# Check for OPENAI_API_KEY environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY environment variable not set."
    read -sp "Please enter your OpenAI API key: " OPENAI_API_KEY
    echo ""
    export OPENAI_API_KEY=$OPENAI_API_KEY
fi

# Activate virtual environment
source venv/bin/activate

# Get the last command
LAST_COMMAND=$(cat config/last_command.txt)

# Run the same command again
echo "Reconnecting to database with previous settings..."
eval "$LAST_COMMAND"

# Deactivate virtual environment
deactivate 