current_dir=$(basename "$PWD")

if [[ "$current_dir" == "later-system" ]]; then
    cd backend
fi


if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

PORT=8000


LOG_DIR="../.logs"
LOG_FILE="$LOG_DIR/backend.log"

if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

> "$LOG_FILE"

if [[ "$1" == "--prod" ]]; then
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT 2>&1 | tee -a "$LOG_FILE"
else
    python3 -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port $PORT 2>&1 | tee -a "$LOG_FILE"
fi
