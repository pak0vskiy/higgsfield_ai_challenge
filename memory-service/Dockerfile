FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
RUN uv pip install --system -e ".[test]"

# Copy source
COPY src/ ./src/

# Data directory for SQLite volume
RUN mkdir -p /data

ENV PYTHONPATH=/app/src
ENV DATA_DIR=/data

EXPOSE 8080

CMD ["uvicorn", "memory_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
