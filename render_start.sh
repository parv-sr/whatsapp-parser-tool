#!/bin/bash

set -e

if [ "${RUN_CELERY_IN_WEB:-false}" = "true" ]; then
  echo "Starting Celery Worker in web container..."
  celery -A config worker --loglevel=info --concurrency=4 &
else
  echo "Skipping Celery worker in web container (RUN_CELERY_IN_WEB=false)"
fi

echo "Starting Web Server..."
uvicorn config.asgi:application --host 0.0.0.0 --port $PORT
