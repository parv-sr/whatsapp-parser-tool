# WhatsApp Real Estate Parser & RAG System
An intelligent, end-to-end Django application designed to ingest unstructured WhatsApp chat exports (specifically for real estate brokers), extract structured property listings using LLMs, and provide a highly accurate Retrieval-Augmented Generation (RAG) chat interface to query the database.

# Key Features
Robust Ingestion Pipeline: Upload raw .txt WhatsApp chat exports. The system parses timestamps, senders, and cleans raw text.

Multi-Stage Deduplication: Prevents database bloat through 3 layers of deduplication (Pre-LLM hashing, In-batch hashing, and Database-level composite key matching).

AI-Powered Extraction: Utilizes OpenAI's gpt-4o-mini with Pydantic structured outputs to turn messy chat messages into structured data (Location, BHK, Price, Transaction Type, Property Type).

Vector Embeddings: Automatically embeds structured listings using text-embedding-3-small and stores them in PostgreSQL using the pgvector extension.

Intent-Aware Hybrid RAG: * Hard Filtering: Distinguishes between property OFFERS ("Flat for sale") and REQUESTS ("Looking to buy a flat") via strict SQL WHERE clauses before vector search.

Hybrid Search: Merges Semantic (Vector) and Lexical (Keyword) search using Reciprocal Rank Fusion (RRF).

Zero-Cost Reranking: Employs local CPU-based cross-encoder reranking (FlashRank + TinyBERT) for pinpoint semantic accuracy.

Asynchronous Processing: Powered by Celery & Redis to handle large file uploads without blocking the UI. Features a live-updating web UI with runtime logs and progress tracking.

# Tech Stack
Backend: Python 3.11+, Django 5.x

Database: PostgreSQL 15+ with pgvector extension

Caching & Task Broker: Redis, Celery

AI & ML: OpenAI API (GPT-4o-mini, Embeddings), FlashRank (Local Reranking)

Frontend: HTML5, CSS3, Vanilla JS, Marked.js (for markdown rendering)

# System Architecture
1. Ingestion & Extraction (apps.ingestion & apps.preprocessing)
User uploads a .txt file.

File is chunked and filtered (regex removes system messages).

Celery Worker dispatches batches to the LLM.

LLM extracts a structured PropertyListing schema.

Deduplication tracker confirms uniqueness.

Unique chunks are saved to the Database as ListingChunk objects.

2. Retrieval & Generation (apps.core)
User asks a question in the chat UI.

Query intent is analyzed to create strict database filters (e.g., intent="OFFER" AND transaction_type="SALE").

Query is vectorized.

Hybrid Search retrieves top 50 candidates from the pgvector database.

FlashRank re-ranks the 50 candidates down to the top 8 most contextually relevant listings.

Context is injected into an OpenAI prompt.

The answer is streamed token-by-token back to the UI.

# Local Development Setup
1. Prerequisites
Python 3.11 or higher

PostgreSQL installed locally (with pgvector installed and enabled).

Redis server running locally (localhost:6379).

2. Clone the Repository
Bash
git clone https://github.com/parv-sr/whatsapp-parser-tool.git
cd whatsapp-parser-tool
3. Create a Virtual Environment
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
4. Install Dependencies
Bash
pip install -r requirements.txt
⚠️ Windows Users: You must install the Windows-compatible binary for python-magic to parse files correctly. Run:
pip uninstall -y libmagic python-magic python-magic-bin
pip install python-magic-bin

5. Environment Variables
Create a .env file in the root directory and add the following:

Ini, TOML
DEBUG=True
SECRET_KEY=your-django-secret-key

# Database Connection
DATABASE_URL=postgres://user:password@localhost:5432/your_db_name

# Redis / Celery Connection
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
REDIS_URL=redis://localhost:6379/1

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key
6. Database Setup
Log into your PostgreSQL shell (psql) and ensure the vector extension exists:

SQL
CREATE DATABASE your_db_name;
\c your_db_name;
CREATE EXTENSION IF NOT EXISTS vector;
Then run Django migrations:

Bash
python manage.py migrate
7. Run the Application
You will need to run two processes in separate terminal windows.

Terminal 1: Django Server

Bash
python manage.py runserver
Terminal 2: Celery Worker

Bash
# On macOS/Linux
celery -A config worker -l INFO

# On Windows (requires thread pool)
celery -A config worker -l INFO --pool=threads
Access the app at http://127.0.0.1:8000.

#Project Structure
Plaintext
whatsapp-parser-tool/
├── apps/
│   ├── core/           # RAG logic, Chat UI, Intent extraction
│   ├── embeddings/     # OpenAI embedding generation
│   ├── ingestion/      # File uploading, parsing, Celery orchestration
│   ├── preprocessing/  # LLM schema extraction, DB models (ListingChunk)
│   └── users/          # Authentication
├── config/             # Django settings, ASGI/WSGI, Celery config
├── static/             # JS scripts (Progress Poller, etc.)
├── templates/          # HTML templates
└── requirements.txt
# Troubleshooting
File Not Found / Storage Error during upload: Ensure your file permissions are correct. If using a cloud bucket in production, the code safely falls back to streaming chunks directly via .open("rb").

ImportError: failed to find libmagic (Windows): Ensure you followed the note in Step 4. You must uninstall libmagic and install python-magic-bin.

Celery tasks pending but not executing: Ensure Redis is running and the Celery worker terminal is active and connected to the correct Broker URL.

# License
Private / Proprietary
