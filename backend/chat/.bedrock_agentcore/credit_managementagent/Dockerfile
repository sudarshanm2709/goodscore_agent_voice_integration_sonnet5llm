FROM python:3.11-slim

WORKDIR /app

# Environment variables for Python & AgentCore
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCKER_CONTAINER=1 \
    RUN_MODE=agentcore \
    AWS_REGION=ap-south-1 \
    AWS_DEFAULT_REGION=ap-south-1 
    
# Copy requirements and install via standard pip
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir aws-opentelemetry-distro==0.12.2

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

# AgentCore Runtime expects port 8080
EXPOSE 8080

# Copy full application context
COPY . .

# Run application using opentelemetry-instrumentation
CMD ["opentelemetry-instrument", "python", "-m", "server"]
