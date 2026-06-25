# Hugging Face Spaces (Docker SDK) — runs the FastAPI backend on port 7860.
# Same image runs locally (docker build/run) and on Spaces unchanged.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Graded core + web layer + the exported KNN templates (PNG dir is gitignored/absent here).
COPY alpr/ ./alpr/
COPY web/ ./web/
COPY data/templates.npz ./data/templates.npz

ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
