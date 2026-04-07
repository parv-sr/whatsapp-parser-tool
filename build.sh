#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing Rust..."

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

source "$HOME/.cargo/env"

echo "Compiling Rust Parser..."
cd rust_parser/whatsapp-parser
cargo build --release
cd ../..

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Running database migrations..."
python manage.py migrate