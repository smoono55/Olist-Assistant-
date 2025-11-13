# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# load .env from project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "olist_analytics.db"
CANONICAL_PARQUET = DATA_DIR / "canonical_orders.parquet"
FAISS_INDEX = DATA_DIR / "faiss_index.idx"
ROW_TEXTS = DATA_DIR / "row_texts.jsonl"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Create data dir if missing
DATA_DIR.mkdir(parents=True, exist_ok=True)
