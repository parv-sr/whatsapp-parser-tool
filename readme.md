# 🏢 WhatsApp Real Estate Parser & RAG Tool

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Django](https://img.shields.io/badge/Django-5.2-092E20.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)
![Celery](https://img.shields.io/badge/Celery-Redis-37814A.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg)

A robust, AI-powered Django application designed to ingest raw WhatsApp chat exports from real estate broker groups, extract structured property listings, and provide a smart semantic search engine (RAG) to query available properties.

## ✨ Key Features

* **Intelligent Chat Ingestion:** Upload `.txt` or `.zip` WhatsApp chat exports. Automatically parses timestamps, senders, and cleans system messages.
* **Multi-Tier Deduplication:** Prevents duplicate listings using exact-match hashing and pre-LLM similarity checks before hitting the database.
* **LLM-Powered Extraction:** Uses OpenAI (`gpt-4o-mini`) to process unstructured chat blocks into structured schema (BHK, Location, Transaction Type, Property Type, Intent).
* **Vector Embeddings:** Generates semantic embeddings (`text-embedding-3-small`) for properties and stores them locally using PostgreSQL's `pgvector` extension.
* **HNSW Indexing:** High-performance approximate nearest neighbor search enabled natively in the database.
* **LangGraph RAG Engine:** A conversational chat interface that intelligently filters queries, performs hybrid retrieval, and answers user queries with exact source snippets.
* **Asynchronous Processing:** Heavy lifting (LLM batching, embedding generation) is offloaded to background Celery workers.

---

## 🏗 Architecture & Data Flow

The project is modularized into highly focused Django apps:

1. **`apps.ingestion`**: Handles file uploads. Unzips archives, reads chat logs, parses out standard WhatsApp formats, drops encrypted/system messages, and creates `RawMessageChunk`s.
2. **`apps.preprocessing`**: Takes raw chunks and batches them to the OpenAI API. Extracts structured `PropertyListing` entities. Normalizes text (e.g., standardizing "buy" to "SALE", "fully furnished" to "FURNISHED") and creates `ListingChunk`s.
3. **`apps.embeddings`**: Generates high-dimensional vector embeddings for the extracted listings and manages the `pgvector` database operations.
4. **`apps.core`**: Houses the main Retrieval-Augmented Generation (RAG) graph and chat views. Parses user queries, extracts hard filters (e.g., "SALE", "RESIDENTIAL"), and queries the vector store to build prompt contexts.
5. **`apps.users`**: Handles authentication and session management.

---

## 🛠 Tech Stack

* **Backend Framework:** Django 5.2.8
* **Database:** PostgreSQL 15+ 
* **Vector Store:** `pgvector` (PostgreSQL Extension)
* **Background Tasks:** Celery + Redis
* **AI & LLMs:** OpenAI API (`gpt-4o-mini` for extraction/chat, `text-embedding-3-small` for vectors)
* **Graph Logic:** LangGraph (for complex RAG routing)
* **Hosting / Deployment:** Render (configured via `build.sh` and `render_start.sh`) / WhiteNoise for static files.

---

## 🚀 Local Development Setup

### 1. Prerequisites
* Python 3.11+
* PostgreSQL (with `pgvector` extension installed)
* Redis Server (for Celery broker/backend)

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd whatsapp-parser-tool
```


### 3. Install Dependencies
Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Database Setup
Ensure PostgreSQL is running and create the database. Crucially, you must enable the vector extension.
```sql
CREATE DATABASE "whatsapp-parser-tool-db";
\c "whatsapp-parser-tool-db"
CREATE EXTENSION vector;
```

### 5. Environment Variables
Create a .env file in the root directory and populate it with your local credentials:

Code snippet
```python
DEBUG=True
DJANGO_SECRET_KEY=your-super-secret-local-key
OPENAI_API_KEY=sk-your-openai-api-key
```

# Database
DB_NAME=whatsapp-parser-tool-db
DB_USER=postgres
DB_PASS=admin@2025
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_URL=redis://127.0.0.1:6379/1

### 6. Run Migrations
Apply the database migrations (this will set up the tables and the HNSW indexes for pgvector):


python manage.py migrate
(Note: If you encounter caching errors, run python manage.py createcachetable to setup the database cache backend).

### 7. Start the Services
You need three terminal windows to run the full stack locally:

#### Terminal 1: Django Server

```bash
python manage.py runserver
```

#### Terminal 2: Redis Server

redis-server

#### Terminal 3: Celery Worker

```bash
celery -A config worker -l info --concurrency=4
```

### 🧪 Running Tests
The project uses Django's standard TestCase combined with unittest.mock to mock OpenAI API calls to prevent billing charges during CI/CD.

To run the test suite:
```bash
python manage.py test
```
Note: The test suite includes custom tearDown methods to safely close dangling asynchronous database connections created by LangGraph/asyncio, ensuring smooth teardowns of the test databases.

### 🚢 Deployment
The project is configured for deployment on platforms like Render or Vercel.

Production Configurations included:

whitenoise for compressed, manifest-based static file serving.
SSL forced redirects and secure cookies (SECURE_SSL_REDIRECT = True).
Celery configured to use the Database URL as the broker with SSL mode enabled.
Custom build.sh script to automate installation, static collection, and migrations during build pipelines.

## To deploy:

Connect your GitHub repository to your hosting provider.
Set the Build Command: ./build.sh
Set the Start Command: ./render_start.sh (or gunicorn config.wsgi:application)
Add all environment variables from your .env to the provider's settings.

### 🔒 Security & Privacy
Uploaded files are processed securely.

The system is configured to intentionally drop System: Messages and calls are end-to-end encrypted. rows and focuses strictly on property listings.

User authentication is required to access the /core/chat/ RAG interface.