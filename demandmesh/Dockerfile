# syntax=docker/dockerfile:1
# Multi-stage build: dependencies -> reproducible training -> slim runtime.
ARG PYTHON_VERSION=3.12

# ---- base: dependencies only (cached unless requirements change) ------------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src

# ---- trainer: generates data and trains the model (fixed seed) -------------
FROM base AS trainer
RUN python -m demandmesh.pipeline --skip-report

# ---- runtime: API image carrying only the artefacts it serves ---------------
FROM base AS runtime
RUN useradd --create-home --uid 10001 appuser
COPY --from=trainer /app/data/processed ./data/processed
COPY --from=trainer /app/data/raw/zones.csv ./data/raw/zones.csv
COPY --from=trainer /app/models ./models
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "demandmesh.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
