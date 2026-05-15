# SHL Assessment Recommender - Conversational Agent

A conversational AI agent that helps hiring managers and recruiters find the right SHL Individual Test Solutions through natural dialogue. Built with FastAPI, Google Gemini, and FAISS semantic search.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Service](#running-the-service)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Design Choices & Trade-offs](#design-choices--trade-offs)

---

## Architecture Overview

```
User Message
     |
     v
+------------------+
|   FastAPI Server  |  (main.py)
|   POST /chat      |
+--------+---------+
         |
         v
+------------------+
|  Query Extractor  |  (agent.py)
|  - tech keywords  |
|  - role keywords   |
|  - assessment types|
+--------+---------+
         |
         v
+------------------+
| Hybrid Retrieval  |  (retrieval.py)
| - FAISS semantic  |
| - Keyword boost   |
| - Multi-query     |
+--------+---------+
         |
         v
+------------------+
|  Gemini LLM Agent |  (agent.py)
|  - System prompt   |
|  - Catalog context |
|  - JSON output     |
+--------+---------+
         |
         v
+------------------+
|  URL Validation   |  (agent.py)
|  - Catalog-only   |
|  - Deduplicate    |
|  - Cap at 10      |
+--------+---------+
         |
         v
  JSON Response
  {reply, recommendations[], end_of_conversation}
```

## Project Structure

```
shl-catalog/
|-- main.py              # FastAPI service with /health and /chat endpoints
|-- agent.py             # Conversational agent (Gemini LLM + prompt engineering)
|-- retrieval.py         # FAISS semantic search + keyword boosting
|-- catalog.py           # Catalog data layer and utilities
|-- scraper.py           # SHL website scraper (fetches catalog listing pages)
|-- enrich_catalog.py    # Enrichment script (fetches descriptions from detail pages)
|-- evaluate.py          # Evaluation harness (schema, behavior probes, recall@10)
|-- chat_cli.py          # Interactive CLI for manual testing
|-- shl_catalog.json     # Scraped catalog data (374 assessments with descriptions)
|-- requirements.txt     # Python dependencies
|-- Dockerfile           # Docker containerization
|-- render.yaml          # Render.com deployment config
|-- .env.example         # Environment variable template
|-- .gitignore           # Git ignore rules
|-- faiss_index/         # Pre-built FAISS index (auto-generated)
```

## Setup & Installation

### Prerequisites

- Python 3.11+
- A Google Gemini API key (free tier works, get one at https://aistudio.google.com/apikey)

### Step 1: Clone and install dependencies

```bash
cd "shl catalog"
pip install -r requirements.txt
```

### Step 2: Set up your API key

```bash
# Copy the example and add your key
cp .env.example .env
```

Edit `.env` and replace with your actual key:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

You can optionally set a model override:
```
GEMINI_MODEL=gemini-3.1-flash-lite
```

Available models (depending on your quota):
| Model | Free Tier RPM | Free Tier RPD | Notes |
|-------|--------------|--------------|-------|
| `gemini-3.1-flash-lite` | 15 | 500 | **Recommended** - highest free quota |
| `gemini-2.5-flash` | 5 | 20 | Good quality, lower quota |
| `gemini-2.5-flash-lite` | 10 | 20 | Balanced |
| `gemini-2.0-flash` | 0 | 0 | May be deprecated on free tier |

### Step 3: Build the FAISS index

```bash
python retrieval.py
```

This encodes all 374 catalog items into embeddings using `all-MiniLM-L6-v2` and saves the FAISS index to `faiss_index/`. Takes ~10 seconds. The index is auto-built on first server startup if missing.

## Running the Service

### Option 1: Run directly

```bash
python main.py
```

Server starts at `http://localhost:8000`. The FAISS index and embedding model load on startup.

### Option 2: Run with uvicorn (production)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option 3: Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

### Verify it's running

```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

### POST /chat

Stateless conversational endpoint. Send the full conversation history each time.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/", "test_type": "K"},
    {"name": "OPQ Leadership Report", "url": "https://www.shl.com/products/product-catalog/view/opq-leadership-report/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Fields:**
- `reply` (string): The agent's conversational response
- `recommendations` (array): Empty `[]` when gathering context or refusing. 1-10 items when ready.
  - `name`: Assessment name (exact match from catalog)
  - `url`: Assessment URL (exact match from catalog)
  - `test_type`: Primary type code (K/P/A/S/C/B/D/E)
- `end_of_conversation` (bool): `true` only when the user is satisfied

### Test type codes

| Code | Meaning | Examples |
|------|---------|---------|
| K | Knowledge & Skills | Java, Python, SQL, AWS, Excel |
| P | Personality & Behavior | OPQ32r, Leadership Report |
| A | Ability & Aptitude | Numerical, Verbal, Inductive reasoning |
| S | Simulation | Phone simulation, Inbox exercise |
| C | Competency | Behavioral competency matching |
| B | Biodata & SJT | Situational judgment tests |
| D | Development | 360 feedback, development reports |
| E | Assessment Experience | Assessment center exercises |

## How It Works

### 1. Catalog Data (`shl_catalog.json`)

The catalog contains 374 SHL Individual Test Solutions scraped from the SHL product catalog website. Each item has:
- **name**: Assessment name
- **url**: Canonical catalog URL
- **test_type**: Array of type codes (K, P, A, S, C, B, D, E)
- **remote_testing**: Whether remote proctoring is available
- **adaptive_irt**: Whether adaptive/IRT scoring is used
- **description**: Detailed description (fetched from detail pages, 365/374 enriched)

**To re-scrape the catalog:**
```bash
python scraper.py          # Scrapes listing pages
python enrich_catalog.py   # Fetches descriptions from detail pages
python retrieval.py        # Rebuilds the FAISS index
```

### 2. Retrieval (`retrieval.py`)

Uses a **hybrid semantic + keyword search** approach:

1. **Semantic search**: User messages are encoded with `all-MiniLM-L6-v2` sentence transformer and matched against catalog embeddings using FAISS (inner product / cosine similarity).

2. **Keyword boosting**: Adds score bonuses when query tokens match:
   - Assessment names (+0.15 per match)
   - Description content (+0.05 per match)
   - Test type keywords (+0.10 when query mentions "personality" and item is type P, etc.)

3. **Multi-query fusion**: Extracts multiple search queries from conversation:
   - Combined user context
   - Latest message
   - Extracted tech keywords (java, python, sql...)
   - Extracted role keywords (developer, manager...)
   - Assessment type keywords (personality, reasoning...)

   Results from all queries are merged by max score.

### 3. Agent (`agent.py`)

The Gemini LLM is prompted with:
- **System prompt**: Rules for clarify/recommend/refine/compare behaviors, test type matching guidelines, JSON output format
- **Catalog context**: Top 20 retrieved assessments with names, URLs, types, descriptions
- **Conversation history**: Full message history from the request

**Post-processing:**
- JSON extraction from LLM output (handles markdown fences, noise)
- URL validation against the catalog (rejects any URL not in `shl_catalog.json`)
- Deduplication and capping at 10 recommendations
- Rate limit retry with exponential backoff (3 attempts)

### 4. Conversational Behaviors

| Behavior | Trigger | Agent Action |
|----------|---------|-------------|
| **Clarify** | Vague query ("I need an assessment") | Asks 1-2 focused questions about role, skills, level |
| **Recommend** | Enough context (role + skills/level) | Returns 1-10 assessments with names and URLs |
| **Refine** | User changes constraints ("add personality tests") | Updates shortlist, does not restart |
| **Compare** | User asks about differences | Compares using catalog data only |
| **Refuse** | Off-topic, legal, salary, prompt injection | Politely declines and redirects |

## Evaluation

### Run the evaluation harness

```bash
# Make sure the server is running first
python main.py

# In another terminal
python evaluate.py
```

**What it tests:**

1. **Health check**: `GET /health` returns 200 with `{"status": "ok"}`

2. **Schema compliance**: Every response has `reply` (string), `recommendations` (array), `end_of_conversation` (bool). Every recommendation has `name`, `url`, `test_type`. All URLs start with `https://www.shl.com/`.

3. **Behavior probes** (6 tests):
   - Refuses off-topic legal questions (no recommendations)
   - Refuses off-topic salary questions (no recommendations)
   - Clarifies vague queries (asks a question, no recommendations)
   - Recommends when given enough context (returns recommendations)
   - Handles refinement mid-conversation (updates recommendations)
   - Refuses prompt injection attempts (no recommendations)

4. **Conversation traces** (if `traces/` directory exists): Replays multi-turn conversations and computes Recall@10.

### Interactive testing

```bash
# Start the server
python main.py

# In another terminal, start the chat CLI
python chat_cli.py
```

Then type naturally:
```
You: I'm hiring a Java developer
Bot: I can help! What seniority level and any specific skills?
You: Mid-level, needs Java, SQL, and good communication
Bot: Here are my recommendations... [shows 5 assessments]
You: Also add some personality tests
Bot: Updated! [shows refined list with personality assessments added]
```

### Test with curl

```bash
# Vague query (should clarify)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need an assessment"}]}'

# Specific query (should recommend)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need a Java programming test for a mid-level developer"}]}'
```

## Deployment

### Deploy to Render

1. Push your code to a GitHub repository
2. Go to [Render](https://render.com) and create a new Web Service
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and configure:
   - **Build command**: `pip install -r requirements.txt && python -c "from retrieval import build_index; build_index()"`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GEMINI_API_KEY` = your key
6. Deploy

Your public URL will be: `https://shl-recommender.onrender.com`

### Deploy with Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

### Deploy to Railway / Fly.io / HuggingFace Spaces

The app is a standard FastAPI service — it works on any platform that supports Python web apps. Just ensure:
1. `GEMINI_API_KEY` is set as an environment variable
2. The build step runs `pip install -r requirements.txt`
3. The start command is `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Design Choices & Trade-offs

### Why Gemini?
- Free tier available (important for a take-home assignment)
- Fast inference (~2-3 seconds per turn)
- Good instruction following for structured JSON output
- `gemini-3.1-flash-lite` offers 500 requests/day on free tier

### Why FAISS + Sentence Transformers?
- `all-MiniLM-L6-v2` is lightweight (~80MB) and fast to encode
- FAISS provides sub-millisecond similarity search over 374 items
- No external vector DB dependency — runs entirely in-process
- Index pre-built at Docker build time for fast cold starts

### Why Hybrid Search?
- Pure semantic search can miss exact keyword matches (e.g., "Java" matching "Coffee")
- Pure keyword search can miss semantic intent (e.g., "coding test" should match "Programming Assessment")
- Hybrid approach: semantic for broad matching + keyword bonus for precision

### Why Stateless API?
- Assignment requirement: every `POST /chat` carries full conversation history
- Simpler to deploy (no session storage, no database)
- Scales horizontally without sticky sessions

### What Didn't Work
- **Gemini 2.0 Flash**: Free tier quota dropped to 0 RPD, had to switch to 3.1 Flash Lite
- **Pure semantic search**: Missed obvious keyword matches for tech skills
- **Over-clarification**: Early prompt versions asked 3-4 questions before recommending, wasting turns. Tuned to 1-2 rounds max.
- **Raw description scraping**: Some detail pages had inconsistent HTML, so 9/374 items have no description

### Rate Limit Handling
- Retry with exponential backoff (10s, 20s, 30s)
- Graceful fallback response on persistent failure (valid schema, no recommendations)
- Configurable model via `GEMINI_MODEL` env var to switch if quota is exhausted
