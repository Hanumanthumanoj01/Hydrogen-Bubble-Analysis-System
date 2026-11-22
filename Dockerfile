# =============================================================================
# Dockerfile — the whole Hydrogen Bubble Analysis System in ONE container
# =============================================================================
# WHERE IT LIVES: in the project folder, next to backend/ and frontend/.
#
# WHAT IT DOES (two stages):
#   1. builds the React frontend (npm run build) into static web files;
#   2. installs the Python backend and copies those files into backend/static/,
#      so FastAPI serves the web application AND the API from one address.
#      The frontend calls the API with relative paths ("/analyze", ...), so
#      no frontend code has to change.
#
# HOW TO RUN IT
#   On Render.com:   nothing to type; Render builds this file itself
#                    (see DEPLOY_GUIDE.pdf).
#   On your own Mac (only if Docker Desktop is installed, optional):
#       docker build -t bubble-app .
#       docker run -p 10000:10000 bubble-app
#       then open http://localhost:10000
#
# SAFETY NOTES
#   - Recordings and results (session_*/, results_*/, videos/) are excluded by
#     .dockerignore, so they are never uploaded into the image.
#   - The memory settings below keep the server under 512 MB (measured peak
#     317 MB for an image, a screen capture and a video in a row).
# =============================================================================

# ---- stage 1: build the web front end ---------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY frontend/ ./
RUN npm run build

# ---- stage 2: Python backend that also serves the built front end ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=10000
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=web /web/dist ./static
EXPOSE 10000
# Render sets $PORT; locally it defaults to 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
