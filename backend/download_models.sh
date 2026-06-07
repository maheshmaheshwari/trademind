#!/bin/bash
# TradeMind AI — Model Download Script
#
# Downloads pre-trained models from cloud storage on first deploy.
# Set MODEL_BUCKET env var to your S3/GCS bucket URL.
#
# Usage:
#   MODEL_BUCKET=s3://your-bucket/trademind-models bash download_models.sh
#   MODEL_BUCKET=gs://your-bucket/trademind-models bash download_models.sh
#   MODEL_URL=https://your-cdn.com/models.tar.gz bash download_models.sh

set -e

FINAL_DIR="$(dirname "$0")/final_models"
mkdir -p "$FINAL_DIR"

MODEL_COUNT=$(ls "$FINAL_DIR"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -gt 300 ]; then
  echo "✅ Models already present ($MODEL_COUNT files) — skipping download"
  exit 0
fi

echo "📦 Downloading models..."

if [ -n "$MODEL_URL" ]; then
  # Direct tarball URL (e.g. from Cloudflare R2, DigitalOcean Spaces, or any CDN)
  curl -L "$MODEL_URL" | tar -xz -C "$FINAL_DIR" --strip-components=1
  echo "✅ Downloaded from $MODEL_URL"

elif [ -n "$MODEL_BUCKET" ]; then
  if [[ "$MODEL_BUCKET" == s3://* ]]; then
    aws s3 sync "$MODEL_BUCKET" "$FINAL_DIR" --exclude "*" --include "*.NS_final.pkl"
    echo "✅ Synced from S3: $MODEL_BUCKET"
  elif [[ "$MODEL_BUCKET" == gs://* ]]; then
    gsutil -m rsync -r "$MODEL_BUCKET" "$FINAL_DIR"
    echo "✅ Synced from GCS: $MODEL_BUCKET"
  else
    echo "❌ Unknown bucket scheme. Use s3:// or gs://"
    exit 1
  fi

else
  echo "❌ No model source configured."
  echo ""
  echo "Set one of:"
  echo "  MODEL_URL=https://your-cdn.com/models.tar.gz"
  echo "  MODEL_BUCKET=s3://your-bucket/trademind-models"
  echo "  MODEL_BUCKET=gs://your-bucket/trademind-models"
  echo ""
  echo "To upload your local models to S3:"
  echo "  aws s3 sync backend/final_models/ s3://your-bucket/trademind-models/ --exclude '*' --include '*.NS_final.pkl'"
  exit 1
fi

MODEL_COUNT=$(ls "$FINAL_DIR"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "✅ $MODEL_COUNT models ready in $FINAL_DIR"
