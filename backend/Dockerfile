# ---- base python image (CPU) ----
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (build tools help with some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code + model
COPY . /app/

# Cloud Run (and Docker) will use $PORT; default to 8080
ENV PORT=8080
EXPOSE 8080

# Optionally set CORS origin at runtime:
# ENV FRONTEND_ORIGIN="http://localhost:3000"

# Start the app (your main.py already runs uvicorn)
CMD ["python", "main.py"]
