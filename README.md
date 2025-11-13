Olist Assistant – Intelligent E-Commerce Analytics & Chat Assistant

A full-stack AI assistant built for exploring the Olist Brazilian E-Commerce Dataset, answering analytical questions, providing recommendations, translating category names, retrieving customer review insights, and optionally searching the web using SerpAPI.

Powered by Python, Fast LLM inference (OpenAI GPT models), RAG-style dataset querying, Gradio UI, and SQLite analytics.
Key Features
1. Dataset-Powered Q&A

The assistant can answer complex analytical questions entirely using your local Olist database, including:

Top products sold

Most sold product per city

Most expensive product per city

Average order value (AOV) for any category & year

Category listings

Cheapest product (global or per city)

Keyword-based product search

Year-based comparisons

Per-city performance

Revenue analysis

2. Smart Intent Recognition

User messages are interpreted using a hybrid rule-based + LLM intent classifier:

data_query

recommend

translate

small_talk

web_search

unknown

This ensures correct routing—dataset queries go to SQL, general questions go to SerpAPI, and casual chat stays conversational.

3. Product Recommendations Engine

Understands natural language requests like:

“What’s good for fiber?”

“Recommend something for eyesight”

“Suggest budget products”

Uses:

Unit sales

Fuzzy name matching

Token scoring

Category relevance

Optional review score weighting

4. Optional Web Search (SerpAPI)

If the dataset lacks information, the assistant can perform real-time Google queries using SerpAPI and summarize results using GPT.

5. Translation of Non-English Category Tokens

Translates only the tokens that appear in the chat, not the entire database.

Example:

cama_mesa_banho → bed, bath & linen
eletronicos → electronics

Uses:

Pattern detection

AUTO_TRANSLATIONS map

Context-aware searching inside chatbox

6. Customer Reviews + Sentiment Summary

If your dataset includes any reviews table (auto-detected), the assistant:

Fetches product reviews

Computes weighted average review score

Summarizes context using LLM

Works automatically with any of:

olist_order_reviews

order_reviews

reviews

order_review

7. Chat Interface (Gradio)

Comes with a clean WhatsApp-style UI:

Left: chat messages

Right: evidence table + translation panel

Input box with Send, Translate, Clear

Automatic conversation memory

8. Context Memory

Stores:

Last categories fetched

Last results

Conversation history

Stored in data/session_memory.json.

9. Automated Testing

Includes two test suites:

Dataset Tests — Ensures all dataset queries work

SerpAPI Tests — Validates live web search + summarization (if API key available)

📁 Project Structure
olist-assistant/
│
├── src/
│   ├── app.py                 # Main Olist Assistant engine + UI
│
├── data/
│   ├── olist_analytics.db     # SQLite database (created during ETL)
│   ├── session_memory.json    # Persistent conversation memory
│
├── tests/
│   ├── test_dataset_queries.py
│   ├── test_serpapi_full.py   # Web search validation
│
├── README.md
└── requirements.txt

⚙️ Installation & Setup
1️⃣ Clone the repo
git clone https://github.com/<your-username>/olist-assistant.git
cd olist-assistant

2️⃣ Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate


(On MacOS/Linux: source .venv/bin/activate)

3️⃣ Install requirements
pip install -r requirements.txt

4️⃣ Set environment variables
PowerShell:
$env:OPENAI_API_KEY="your-openai-key"
$env:SERPAPI_API_KEY="your-serpapi-key"   # optional

Bash/Mac/Linux:
export OPENAI_API_KEY="your-openai-key"
export SERPAPI_API_KEY="your-serpapi-key"

5️⃣ Build the SQLite database

Place your processed Olist database at:

data/olist_analytics.db


Or generate it using your ETL pipeline.

6️⃣ Run the app
python -u src/app.py


The UI opens at:

http://127.0.0.1:7860

Running Tests
Dataset Query Tests
python tests/test_dataset_queries.py

SerpAPI Web Search Tests
python tests/test_serpapi_full.py

Example Questions the Assistant Can Answer
Dataset Questions

✔ “Top 5 cities by revenue in 2017”
✔ “Most sold product per city”
✔ “Cheapest product”
✔ “Average order value for eletronicos in 2018”
✔ “List top 10 expensive products”
✔ “Reviews of the most sold product”

Recommendation Questions

✔ “What’s good for fiber?”
✔ “Suggest something for eyesight”
✔ “Show me budget options for electronics”

Web Search Questions

✔ “Who is Jeff Bezos?”
✔ “Latest news about OpenAI”
✔ “What is quantum computing?”

Troubleshooting
Environment variables not detected?

Run:

Get-ChildItem Env:

Chatbot doesn’t respond?

Look for errors in:

console output

Web search not working?

Ensure:

SERPAPI_API_KEY is set
SerpAPI quota is available

SQLite errors?

Ensure olist_analytics.db contains required tables:

orders_items (mandatory)

olist_order_reviews or similar (optional)

🏁 Final Notes

✔ Zero breaking changes to the core logic
✔ Fully modular and ready for deployment
✔ Works offline (dataset queries) + online (SerpAPI search)
✔ Dynamic, scalable, and easily extensible
