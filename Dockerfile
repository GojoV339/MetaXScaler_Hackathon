FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create data directory if not present
RUN mkdir -p env/data

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Environment variables (will be overridden by HF Secrets)
ENV API_BASE_URL="https://router.huggingface.co/v1"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server
CMD ["python", "app.py"]
