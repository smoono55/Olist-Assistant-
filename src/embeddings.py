# src/embeddings.py
"""
Build Sentence-Transformers embeddings for each canonical row and write a FAISS index + metadata.

- Reads parquet if available; falls back to CSV.
- Uses GPU if torch.cuda.is_available() else CPU.
- Saves:
    data/faiss_index.idx
    data/row_texts.jsonl   (one json per line with keys: i, order_id, text)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
import os
import sys

# optional: prefer a model name from env or default
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH", 128))
TOPK = 6
SAMPLE_LIMIT = int(os.environ.get("EMBED_SAMPLE_LIMIT", 0))  # 0 means encode all rows
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CANONICAL_PARQUET = DATA_DIR / "canonical_orders.parquet"
CANONICAL_CSV = DATA_DIR / "canonical_orders.csv"
FAISS_INDEX = DATA_DIR / "faiss_index.idx"
ROW_TEXTS = DATA_DIR / "row_texts.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_canonical():
    if CANONICAL_PARQUET.exists():
        print("[emb] loading parquet:", CANONICAL_PARQUET)
        df = pd.read_parquet(CANONICAL_PARQUET)
    elif CANONICAL_CSV.exists():
        print("[emb] parquet not found; loading CSV:", CANONICAL_CSV)
        df = pd.read_csv(CANONICAL_CSV, low_memory=False)
    else:
        raise FileNotFoundError("No canonical file found. Run ETL first to produce canonical_orders.parquet or .csv in data/")
    return df

def make_row_texts(df):
    texts = []
    for i, r in df.reset_index().iterrows():
        order_id = r.get("order_id") or ""
        date = str(r.get("order_purchase_timestamp") or r.get("order_year") or "")
        city = r.get("customer_city") or r.get("geo_city") or ""
        cat = r.get("product_category_name") or ""
        price = r.get("price") if "price" in r else ""
        text = f"order_id:{order_id} date:{date} city:{city} category:{cat} price:{price}"
        texts.append({"i": int(i), "order_id": order_id, "text": text})
    return texts

def encode_and_index(texts, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    # lazy import to keep startup light
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[emb] using model '{model_name}' on device: {device}")

    model = SentenceTransformer(model_name, device=device)
    corpus = [t["text"] for t in texts]
    n = len(corpus)
    if SAMPLE_LIMIT and SAMPLE_LIMIT > 0:
        print(f"[emb] SAMPLE_LIMIT={SAMPLE_LIMIT} -> encoding only first {SAMPLE_LIMIT} rows for dev")
        corpus = corpus[:SAMPLE_LIMIT]
        texts = texts[:SAMPLE_LIMIT]
        n = len(corpus)

    # encode in batches
    print(f"[emb] encoding {n} rows with batch_size={batch_size} ...")
    embeddings = model.encode(corpus, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    d = embeddings.shape[1]
    print(f"[emb] embeddings shape: {embeddings.shape}")

    # build faiss index
    print("[emb] building FAISS IndexFlatL2 ...")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(f"[emb] FAISS index total vectors: {index.ntotal}")

    # write index
    faiss.write_index(index, str(FAISS_INDEX))
    print("[emb] FAISS index written to", FAISS_INDEX)

    # write metadata (texts)
    with open(ROW_TEXTS, "w", encoding="utf8") as fh:
        for t in texts:
            fh.write(json.dumps(t, default=str, ensure_ascii=False) + "\n")
    print("[emb] metadata written to", ROW_TEXTS)

def main():
    df = load_canonical()
    texts = make_row_texts(df)
    encode_and_index(texts)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[emb] ERROR:", e)
        import traceback; traceback.print_exc()
        sys.exit(1)
