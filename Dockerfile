# ─────────────────────────────────────────────────────────────────────────────
# Sudoku CV Solver — Docker image
# ─────────────────────────────────────────────────────────────────────────────
# Base: Python 3.11 + Playwright Chromium + OpenCV + PyTorch (CPU)
#
# Build:
#   docker build -t sudoku-cv-solver .
#
# Run (with display for non-headless mode):
#   docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#              sudoku-cv-solver python src/main.py
#
# Run headless (CI / server):
#   docker run --rm sudoku-cv-solver python src/main.py --headless
#
# Run benchmark:
#   docker run --rm sudoku-cv-solver python src/main.py --benchmark --headless
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV headless requirements
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # General utils
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Playwright Chromium ──────────────────────────────────────────────────────
RUN playwright install chromium --with-deps

# ─── Application code ─────────────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/templates/ ./data/templates/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default: run the full pipeline headless with EasyOCR
CMD ["python", "src/main.py", "--headless", "--digit-method", "easyocr"]
