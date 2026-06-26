cd "$(cd "$(dirname "$0")/.." && pwd)"
source venv/bin/activate

echo "Stopping Uvicorn server..."
pkill -f "uvicorn api.server:app" || true

echo "Wiping corrupted database..."
rm -f nifty500.db*

echo "Running data pipeline..."
python scripts/run_pipeline.py > pipeline.log 2>&1

echo "Pipeline finished. Starting Uvicorn server back up..."
nohup uvicorn api.server:app --port 8000 > server.log 2>&1 &
echo "Done!"
