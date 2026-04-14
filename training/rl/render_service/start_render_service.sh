#!/bin/bash
# Start HTML Render Service for RL Training
#
# This script starts the Playwright-based HTML rendering service
# that provides sandboxed rendering for reward computation.
#
# Usage:
#   bash start_render_service.sh
#
# Environment Variables:
#   RENDER_SERVICE_HOST: Listen host (default: 0.0.0.0)
#   RENDER_SERVICE_PORT: Listen port (default: 8768)
#   RENDER_SERVICE_WORKERS: Number of workers (default: 256)
#   RENDER_SERVICE_AUTO_SHUTDOWN: Enable auto-shutdown (default: true)
#   RENDER_SERVICE_IDLE_TIMEOUT: Idle timeout in seconds (default: 600)

set -x

# Configuration
RENDER_SERVICE_HOST=${RENDER_SERVICE_HOST:-"0.0.0.0"}
RENDER_SERVICE_PORT=${RENDER_SERVICE_PORT:-8768}
RENDER_SERVICE_WORKERS=${RENDER_SERVICE_WORKERS:-256}
LOG_LEVEL=${LOG_LEVEL:-"info"}

# Auto-shutdown settings
export RENDER_SERVICE_AUTO_SHUTDOWN=${RENDER_SERVICE_AUTO_SHUTDOWN:-"true"}
export RENDER_SERVICE_IDLE_TIMEOUT=${RENDER_SERVICE_IDLE_TIMEOUT:-600}

echo "========================================"
echo "HTML Render Service Configuration"
echo "========================================"
echo "Host: ${RENDER_SERVICE_HOST}"
echo "Port: ${RENDER_SERVICE_PORT}"
echo "Workers: ${RENDER_SERVICE_WORKERS}"
echo "Auto-shutdown: ${RENDER_SERVICE_AUTO_SHUTDOWN}"
echo "Idle timeout: ${RENDER_SERVICE_IDLE_TIMEOUT}s"
echo "========================================"

# Check Python environment
echo ""
echo "Checking environment..."
python3 --version

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import fastapi; print('FastAPI:', fastapi.__version__)" || {
    echo "Installing FastAPI..."
    pip install fastapi
}
python3 -c "import uvicorn; print('Uvicorn:', uvicorn.__version__)" || {
    echo "Installing Uvicorn..."
    pip install uvicorn[standard]
}
python3 -c "import playwright; print('Playwright: installed')" || {
    echo "Installing Playwright..."
    pip install playwright
    playwright install chromium
}

# Start service
echo ""
echo "Starting HTML Render Service..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m uvicorn html_render_service:app \
    --host ${RENDER_SERVICE_HOST} \
    --port ${RENDER_SERVICE_PORT} \
    --workers ${RENDER_SERVICE_WORKERS} \
    --log-level ${LOG_LEVEL} \
    --access-log

echo ""
echo "Service stopped."
