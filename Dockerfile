# 🐳 ML Galaxy Portfolio - FastAPI Service Dockerfile (2026 Edition)
# Sets up a lightweight, containerized environment for microservice serving.

# Multi-stage lightweight base
FROM python:3.11-slim as builder

# Set workspace directory
WORKDIR /app

# Prevent python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install systems tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY production/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runner stage
FROM python:3.11-slim as runner

WORKDIR /app

# Copy python dependencies path from builder
COPY --from=builder /root/.local /root/.local
COPY production/ /app/production/

ENV PATH=/root/.local/bin:$PATH

# Expose FastAPI default port
EXPOSE 8000

# Start Uvicorn ASGI server hosting the microservice
CMD ["uvicorn", "production.api_service:app", "--host", "0.0.0.0", "--port", "8000"]
