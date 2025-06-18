# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api_vanna.py .
COPY beast_mode_trainer.py .
COPY frontend/ ./frontend/

# Create necessary directories
RUN mkdir -p vanna_storage/models vanna_storage/connections

# Expose ports
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start the backend API server\n\
python api_vanna.py &\n\
BACKEND_PID=$!\n\
\n\
# Wait for backend to start\n\
sleep 10\n\
\n\
# Start the frontend\n\
cd frontend && streamlit run main.py --server.port 8501 --server.address 0.0.0.0 &\n\
FRONTEND_PID=$!\n\
\n\
# Wait for any process to exit\n\
wait $BACKEND_PID $FRONTEND_PID' > /app/start.sh

RUN chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"] 