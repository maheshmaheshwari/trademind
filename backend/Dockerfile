# TradeMind backend — Hugging Face Docker Space image.
# Deployed with scripts/deploy_space.py (see docs/DEPLOYMENT.md).
FROM python:3.9-slim

# lightgbm needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000 — everything writable lives in /app
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
# CPU-only torch first: the default PyPI wheel bundles CUDA (~5 GB image bloat)
# and the Space has no GPU. requirements.txt's torch pin is then already met.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
# Runtime dirs (ephemeral): models + data are pulled from the encrypted HF Hub
# store at startup by sync_models(); logs/archives are scratch.
RUN mkdir -p logs data final_models model_archives/previous_models .cache \
 && chown -R user:user /app

USER user
ENV HF_HOME=/app/.cache \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
