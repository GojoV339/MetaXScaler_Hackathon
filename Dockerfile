FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Add a non-root user (Hugging Face Spaces run as uid 1000 by default)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Ensure data directory exists
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
