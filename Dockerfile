FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install -e . --no-cache-dir
COPY . .
ENV PYTHONPATH=/app
CMD ["python", "scripts/run_live.py"]
