#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

cargo build --release

cd rust_parser/whatsapp-parser && cargo build --release && \
cd /app && mkdir -p /app/bin && \
cp rust_parser/whatsapp-parser/target/release/whatsapp-parser /app/bin/whatsapp-parser && \
chmod +x /app/bin/whatsapp-parser

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Running database migrations..."
python manage.py migrate