#!/bin/bash

set -e

echo "Starting Celery Worker..."
celery -A config worker --loglevel=info --concurrency=4 &

echo "Starting Web Server..."
uvicorn config.asgi:application --host 0.0.0.0 --port $PORT