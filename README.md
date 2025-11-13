# Olist-Assistant-
A lightweight, Gradio-based assistant for exploring the Olist ecommerce dataset. It combines rule-based SQL lookups, fuzzy product search, simple RAG ideas (dataset evidence + optional SerpAPI web fallback), and small LLM steps for intent classification and answer polishing. Built to be easy to run locally for data exploration, demos, and testing.
Project summary

Olist Assistant is an interactive chat UI that answers questions about product categories, top sellers, average order value (AOV), cheapest/most expensive items, per-city summaries, product recommendations, and customer reviews — all powered primarily by your local olist_analytics.db SQLite dataset. When a user asks something that is not in the dataset, the assistant can optionally query the web (via SerpAPI) and produce a short web summary injected into the chat.

Key design goals:

Dataset-first answers (no web-search noise unless asked or needed)

Transparent evidence: SQL rows are shown in the Evidence panel

Simple natural-language intent detection with rule+LLM fallback

Lightweight, replicable local setup (Gradio UI + sqlite DB)

Translate Portuguese category tokens to readable English only on demand (Translate button) and only for tokens present in the chat

Features

Ask conversational questions, e.g.:

what all products are available

Average order value for eletronicos in 2018

Top 5 cities by revenue in 2017

most sold product in each city

cheapest product in Sao Paulo

reviews for <product_id>

what's good for fiber (dataset-backed recommendations)

Evidence panel: displays SQL result snippets

Translate button: translates category tokens found only in the chat (no global dump)

Optional web fallback via SerpAPI; when web results are returned, the assistant adds a short LLM-generated web summary into the chat

Review fetching and scoring (if reviews table exists in DB)

Fuzzy product name search using rapidfuzz

Persistent lightweight session memory (data/session_memory.json) to remember last categories/results

Repo contents (important files)

src/app.py — main application (Gradio UI + agent logic)

data/olist_analytics.db — not included; you must create/provide this (ETL)

data/session_memory.json — auto-created by the app

tests/ — (optional) test scripts you may add for automated checks (SerpAPI tests, intent tests)

Quick start (local)

Clone the repo:

git clone <your-repo-url>
cd olist-agent


Create a Python virtualenv and install:

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or: source .venv/bin/activate  # mac/linux
python -m pip install -r requirements.txt


If requirements.txt is not present, install at least:

pip install gradio openai rapidfuzz requests


Provide required environment variables:

OPENAI_API_KEY — required for intent classification & formatting

SERPAPI_API_KEY — optional, only if you want live web search fallback

Example (PowerShell):

$env:OPENAI_API_KEY = 'sk-...'
$env:SERPAPI_API_KEY = 'your_serpapi_key_here'   # optional


Provide the SQLite DB:

Place olist_analytics.db in data/ (path: data/olist_analytics.db).

The app expects common column names used in Olist ETL outputs (orders_items, optional products and reviews tables). If table/column names differ, adapt SQL or ETL.

Run the app:

python -u src/app.py


Open http://127.0.0.1:7860 in a browser.

Configuration & environment

OPENAI_API_KEY — required

SERPAPI_API_KEY — optional

LLM_CLASSIFY_MODEL, LLM_FORMAT_MODEL — optional environment variables to override the default LLM models (defaults in code: gpt-4o-mini)

data/ — persistent data folder; session memory stored at data/session_memory.json

Example queries & expected behavior

what all products are available
→ Returns list of categories (updates last_categories in memory). No web search.

Average order value for eletronicos in 2018
→ Uses SQL to compute AOV for category/year and replies with numeric result and evidence.

Top 5 cities by revenue in 2017
→ Returns top cities, revenue and orders.

most sold product in each city
→ Dataset-first response listing the top product per city (based on sales counts).

cheapest product in Sao Paulo
→ SQL to find the minimum price product in city.

reviews for <product_id> or reviews for the most sold product
→ Attempts to fetch reviews and compute average score if reviews table exists.

what's good for fiber
→ Dataset-backed recommendations via fuzzy search + scoring; if nothing found and SERPAPI is enabled, optionally searches the web.

Translation behavior (important)

Translate button translates only category-like tokens found inside the chat messages (tokens with underscores or explicit category tokens). It does not show or dump the entire translation dictionary by default.

This prevents the UI from constantly showing the whole translation DB; translations appear only when the user clicks Translate after tokens appear in the chat.

Troubleshooting (common issues you've seen)

ModuleNotFoundError: No module named 'src' when running tests
Run tests from repo root and ensure PYTHONPATH contains the project root. Example:

python -m pytest tests
# or set PYTHONPATH for imports in test files:
$env:PYTHONPATH = (Get-Location).Path
python tests/test_serpapi_full.py


Gradio Chatbot tuple/format errors
Use type="messages" for the Chatbot component and pass messages in the OpenAI-style dict format {"role":"user","content":...} / {"role":"assistant","content":...} as implemented.

Nothing shows in Evidence panel
Ensure your data/olist_analytics.db has the expected orders_items table and column names. If columns differ, adapt src/app.py SQL or your ETL output.

Web search not returning results
Make sure SERPAPI_API_KEY is set and valid. The test script may return empty results if search quota is exhausted or the API key is invalid.

Translations panel shows everything
The current code restricts translations to tokens found in chat messages. If you see more, confirm you are running the latest src/app.py.

Testing

You can add test scripts in tests/ that import functions from src/app.py (ensure tests run from repo root or PYTHONPATH is set). Example tests to include:

Intent classification tests (various user phrasings → expected task)

Dataset query tests (mock DB or small sample DB)

SerpAPI integration tests (requires valid SERPAPI_API_KEY)

Translate button behavior tests (simulate chat messages containing category tokens)

Development notes & extension ideas

Add caching for web-search results and LLM web-summaries to minimize API calls and costs.

Improve product-name matching by indexing product names or using a small vector index.

Add authentication and persistent user sessions for multi-user deployments.

Provide export (CSV/PDF) for evidence and summaries.

License & acknowledgements

This project is provided as-is for educational and demo purposes. Choose an appropriate license (MIT, Apache 2.0, etc.) for your repo.
Thanks to the open-source tools used: Gradio, RapidFuzz, and OpenAI.

Short project blurb (for GitHub project description)

Olist Assistant — conversational dataset assistant for the Olist ecommerce dataset. Ask natural-language questions about categories, top-sellers, AOV, cheapest/most expensive products, per-city statistics, and customer reviews. Dataset-first responses with optional SerpAPI web fallback and short LLM-generated summaries injected into the chat.
