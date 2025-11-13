# src/app.py
"""
Olist Assistant — expanded dataset capabilities + improved UX

Run:
    python -u src/app.py

Requirements:
 - OPENAI_API_KEY in env
 - Optional SERPAPI_API_KEY for web search fallback
 - data/olist_analytics.db must exist
"""

import os
import re
import json
import time
import sqlite3
import threading
import traceback
from typing import List, Dict, Any, Tuple, Optional

import requests
import openai
from rapidfuzz import process, fuzz
import gradio as gr

# -------- configuration ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "olist_analytics.db")
SESSION_MEM_PATH = os.path.join(DATA_DIR, "session_memory.json")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment before running this app.")
openai.api_key = OPENAI_KEY

LLM_CLASSIFY_MODEL = os.getenv("LLM_CLASSIFY_MODEL", "gpt-4o-mini")
LLM_FORMAT_MODEL = os.getenv("LLM_FORMAT_MODEL", "gpt-4o-mini")

MAX_EVIDENCE_ROWS = 12
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860

# -------- DB helpers ----------
def db_connect():
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"Database missing at {DB_PATH}. Run ETL first.")
    return sqlite3.connect(DB_PATH)

def run_sql_fetchall(sql: str, params: tuple = ()):
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()

def table_exists(name: str) -> bool:
    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cur.fetchone() is not None
    finally:
        conn.close()

# -------- memory (persisted) ----------
DEFAULT_MEMORY = {"conversations": [], "last_categories": [], "last_results": []}

def load_memory():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(SESSION_MEM_PATH):
            with open(SESSION_MEM_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return DEFAULT_MEMORY.copy()

def save_memory(mem):
    try:
        with open(SESSION_MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

MEMORY = load_memory()

def memory_add_conversation(user_text: str, assistant_text: str):
    MEMORY.setdefault("conversations", []).append({
        "user": user_text,
        "assistant": assistant_text,
        "ts": int(time.time())
    })
    MEMORY["conversations"] = MEMORY["conversations"][-500:]
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()

# -------- utilities ----------
def fuzzy_match_names(name_list: List[str], keywords: List[str], limit=10):
    hits = {}
    for kw in keywords:
        results = process.extract(kw, name_list, scorer=fuzz.partial_ratio, limit=limit)
        for name, score, _ in results:
            if score >= 65:
                hits[name] = max(hits.get(name, 0), score)
    return sorted(hits.items(), key=lambda x: -x[1])

# -------- translations ----------
AUTO_TRANSLATIONS = {
    "cama_mesa_banho": "bed, bath & linen",
    "beleza_saude": "beauty & health",
    "esporte_lazer": "sports & leisure",
    "moveis_decoracao": "furniture & decor",
    "informatica_acessorios": "computer accessories",
    "utilidades_domesticas": "household goods",
    "relogios_presentes": "watches & gifts",
    "telefonia": "telephony",
    "ferramentas_jardim": "tools & garden",
    "automotivo": "automotive",
    "brinquedos": "toys",
    "cool_stuff": "cool stuff",
    "perfumaria": "perfumery",
    "bebes": "baby products",
    "eletronicos": "electronics",
    "papelaria": "stationery",
    "pet_shop": "pet shop",
    "alimentos": "food",
    "alimentos_bebidas": "food & beverages",
    "bebidas": "drinks",
}

def translate_categories(categories: List[str]) -> Dict[str, str]:
    out = {}
    for c in categories:
        if not c:
            continue
        if c in AUTO_TRANSLATIONS:
            out[c] = AUTO_TRANSLATIONS[c]
        else:
            out[c] = c.replace("_", " ")
    return out

# -------- dataset functions ----------
def list_categories(top_n=60):
    rows = run_sql_fetchall("""
        SELECT product_category_name, COUNT(*) as cnt
        FROM orders_items
        GROUP BY product_category_name
        ORDER BY cnt DESC
        LIMIT ?
    """, (top_n,))
    for r in rows:
        if r.get("product_category_name") is None:
            r["product_category_name"] = "None"
    MEMORY["last_categories"] = [r["product_category_name"] for r in rows]
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
    return rows

def aov_for_category_year(category: str, year: str):
    rows = run_sql_fetchall("""
        SELECT AVG(price) as avg_price, COUNT(*) as n
        FROM orders_items
        WHERE product_category_name = ? AND STRFTIME('%Y', order_purchase_timestamp) = ?
    """, (category, year))
    return rows[0] if rows else {"avg_price": 0, "n": 0}

def top_cities_by_revenue(year: str, topk=5):
    rows = run_sql_fetchall("""
        SELECT customer_city as city, SUM(item_total) as revenue, COUNT(*) as orders
        FROM orders_items
        WHERE STRFTIME('%Y', order_purchase_timestamp)=?
        GROUP BY customer_city
        ORDER BY revenue DESC
        LIMIT ?
    """, (year, topk))
    return rows

def top_products_by_units(topk=10, year: Optional[str] = None):
    if year:
        rows = run_sql_fetchall("""
            SELECT product_id, product_category_name, COUNT(*) as units_sold, SUM(item_total) as revenue
            FROM orders_items
            WHERE STRFTIME('%Y', order_purchase_timestamp)=?
            GROUP BY product_id
            ORDER BY units_sold DESC
            LIMIT ?
        """, (year, topk))
    else:
        rows = run_sql_fetchall("""
            SELECT product_id, product_category_name, COUNT(*) as units_sold, SUM(item_total) as revenue
            FROM orders_items
            GROUP BY product_id
            ORDER BY units_sold DESC
            LIMIT ?
        """, (topk,))
    MEMORY["last_results"] = rows
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
    return rows

def expensive_products(topk=10):
    rows = run_sql_fetchall("""
        SELECT product_id, product_category_name, MAX(price) as max_price
        FROM orders_items
        GROUP BY product_id
        ORDER BY max_price DESC
        LIMIT ?
    """, (topk,))
    MEMORY["last_results"] = rows
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
    return rows

def most_expensive_per_city():
    rows = run_sql_fetchall("""
        SELECT o.customer_city as city, o.product_id, o.product_category_name, o.price
        FROM orders_items o
        JOIN (
            SELECT customer_city, MAX(price) as max_price
            FROM orders_items
            GROUP BY customer_city
        ) m ON m.customer_city = o.customer_city AND m.max_price = o.price
        ORDER BY o.price DESC
    """)
    MEMORY["last_results"] = rows
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
    return rows

def most_sold_product_per_city():
    rows = run_sql_fetchall("""
        SELECT customer_city as city, product_id, product_category_name, COUNT(*) as units_sold
        FROM orders_items
        GROUP BY customer_city, product_id
    """)
    best = {}
    for r in rows:
        city = r.get("city") or "Unknown"
        if city not in best or r.get("units_sold", 0) > best[city].get("units_sold", 0):
            best[city] = r
    result = list(best.values())
    MEMORY["last_results"] = result
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
    return result

def cheapest_product(topk=1, city: Optional[str] = None):
    if city:
        rows = run_sql_fetchall("""
            SELECT product_id, product_category_name, MIN(price) as min_price
            FROM orders_items
            WHERE customer_city = ?
            GROUP BY product_id
            ORDER BY min_price ASC
            LIMIT ?
        """, (city, topk))
    else:
        rows = run_sql_fetchall("""
            SELECT product_id, product_category_name, MIN(price) as min_price
            FROM orders_items
            GROUP BY product_id
            ORDER BY min_price ASC
            LIMIT ?
        """, (topk,))
    MEMORY["last_results"] = rows
    threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
    return rows

def search_products_by_keyword(kw: str, topk=6):
    conn = db_connect()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='olist_products'")
        if cur.fetchone():
            cur.execute("PRAGMA table_info('olist_products')")
            cols = [r[1] for r in cur.fetchall()]
            name_col = next((c for c in cols if "name" in c.lower()), None)
            if name_col:
                cur.execute(f"SELECT product_id, product_category_name, {name_col} as pname FROM olist_products WHERE {name_col} IS NOT NULL LIMIT 20000")
                fetched = cur.fetchall()
                names = [r[2] for r in fetched]
                matches = fuzzy_match_names(names, [kw], limit=50)
                matched_names = [n for n, s in matches][:topk]
                results = []
                for r in fetched:
                    if r[2] in matched_names:
                        results.append({"product_id": r[0], "product_category_name": r[1], "product_name": r[2]})
                for r in results:
                    cur.execute("SELECT COUNT(*) as units_sold, AVG(price) as avg_price FROM orders_items WHERE product_id=?", (r["product_id"],))
                    rr = cur.fetchone()
                    r["units_sold"] = rr[0] if rr else 0
                    r["avg_price"] = rr[1] if rr else None
                MEMORY["last_results"] = results
                threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
                return results
        # fallback: category-based search
        rows = run_sql_fetchall("""
            SELECT product_id, product_category_name, COUNT(*) as units_sold, AVG(price) as avg_price
            FROM orders_items
            WHERE product_category_name LIKE ?
            GROUP BY product_id
            ORDER BY units_sold DESC
            LIMIT ?
        """, (f"%{kw}%", topk))
        MEMORY["last_results"] = rows
        threading.Thread(target=save_memory, args=(MEMORY,), daemon=True).start()
        return rows
    finally:
        conn.close()

# -------- reviews lookup and scoring ----------
def get_review_table_name():
    # try common Olist/etl names
    for candidate in ("olist_order_reviews", "order_reviews", "reviews", "order_review"):
        if table_exists(candidate):
            return candidate
    return None

def get_reviews_for_product(product_id: str, limit=20):
    tbl = get_review_table_name()
    if not tbl:
        return []
    conn = db_connect()
    try:
        cur = conn.cursor()
        # common columns: review_id, product_id, review_score, review_comment_title, review_comment_message
        # try different column names with LIKE
        cols_info = run_sql_fetchall(f"PRAGMA table_info('{tbl}')")
        col_names = [c['name'] for c in cols_info] if cols_info else []
        pid_col = next((c for c in col_names if "product" in c and "id" in c), None)
        score_col = next((c for c in col_names if "score" in c or "rate" in c), None)
        msg_col = next((c for c in col_names if "comment" in c or "message" in c or "review" in c and "text" in c), None)
        if not pid_col:
            return []
        score_col = score_col or "review_score"
        msg_col = msg_col or "review_comment_message"
        sql = f"SELECT {pid_col} as product_id, {score_col} as score, {msg_col} as message FROM {tbl} WHERE {pid_col}=? LIMIT ?"
        cur.execute(sql, (product_id, limit))
        fetched = cur.fetchall()
        out = []
        for row in fetched:
            # row may be tuple; create mapping using column names if possible
            if isinstance(row, tuple) and len(row) >= 3:
                out.append({"product_id": row[0], "score": row[1], "message": row[2]})
            else:
                out.append(dict(row))
        return out
    except Exception:
        return []
    finally:
        conn.close()

def compute_review_score_for_product(product_id: str):
    reviews = get_reviews_for_product(product_id, limit=500)
    if not reviews:
        return None
    scores = [r.get("score") for r in reviews if r.get("score") is not None]
    try:
        scores = [float(s) for s in scores]
    except Exception:
        scores = []
    if not scores:
        return None
    avg = sum(scores) / len(scores)
    return {"avg_score": avg, "n": len(scores)}

# -------- serpapi web search ----------
def serpapi_search(query: str, num=3):
    if not SERPAPI_KEY:
        return []
    try:
        params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": num}
        resp = requests.get("https://serpapi.com/search", params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        results = []
        if "organic_results" in data:
            for r in data["organic_results"][:num]:
                results.append({"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")})
        return results
    except Exception:
        return []

# -------- LLM helpers ----------
def classify_intent_openai(user_text: str) -> str:
    system = ("You are a compact intent classifier. Return exactly one token: small_talk, data_query, recommend, translate, web_search, unknown.")
    prompt = f"Label the user's intent in one token. Message: {user_text}\nReturn the single token only."
    try:
        resp = openai.ChatCompletion.create(
            model=LLM_CLASSIFY_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3
        )
        label = resp.choices[0].message.content.strip().split()[0].lower()
        if label in ("small_talk", "data_query", "recommend", "translate", "web_search", "unknown"):
            return label
    except Exception:
        pass
    txt = user_text.lower()
    if any(w in txt for w in ("hi", "hello", "hey", "how are")):
        return "small_talk"
    if any(w in txt for w in ("recommend", "good for", "suggest", "best for", "what's good")):
        return "recommend"
    if any(w in txt for w in ("translate", "what language", "mean", "english", "translate them")):
        return "translate"
    if any(w in txt for w in ("average", "aov", "average order", "top", "most sold", "most selling", "best sellers", "cities", "revenue", "available", "products", "category", "cheapest", "cheapest product", "review", "reviews")):
        return "data_query"
    if any(w in txt for w in ("who is", "what is", "how to", "latest", "news", "price of", "define")):
        return "web_search"
    return "unknown"

def format_with_llm(user_text: str, short_summary: str, evidence: List[Dict[str, Any]], citations: List[Dict[str, Any]] = []):
    system = "You are a friendly, concise ecommerce assistant. Provide a short (1-3 sentence) answer, mention if dataset was used and offer one short follow-up suggestion."
    ev_preview = ""
    if evidence:
        snippets = []
        for r in evidence[:3]:
            snippets.append(" | ".join(f"{k}={v}" for k, v in list(r.items())[:4]))
        ev_preview = "Evidence sample:\n" + "\n".join(snippets)
    cit_preview = ""
    if citations:
        cit_preview = "\nWeb citations:\n" + "\n".join([f"- {c.get('title','')}: {c.get('link','')}" for c in citations[:3]])
    prompt = f"User: {user_text}\n\nSummary:\n{short_summary}\n\n{ev_preview}\n{cit_preview}\n\nWrite a short reply (1-3 sentences) and add one short suggested follow-up question."
    try:
        resp = openai.ChatCompletion.create(
            model=LLM_FORMAT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=220
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # fallback simple reply
        return short_summary if len(short_summary) < 800 else short_summary[:800] + "..."

# -------- dataset-intent extraction (rules + LLM fallback) ----------
def extract_dataset_intent(user_text: str) -> Dict[str, Any]:
    txt = user_text.lower()
    # rules for richer user intents
    if re.search(r"cheapest product (per|in|for)\s+([a-z0-9_ ]+ city|city|in\s+[a-z ]+)", txt):
        # try city name
        m = re.search(r"in\s+([a-z ]+)$", txt)
        city = m.group(1).strip() if m else None
        return {"task": "cheapest", "params": {"city": city}}
    if re.search(r"cheapest product|what is the cheapest product|cheapest item", txt):
        return {"task": "cheapest", "params": {}}
    if re.search(r"review(s)? (of|for|on)|customer review|what do customers say|review score", txt):
        # could be asking about product - try to extract product id or 'most sold' phrasing
        m = re.search(r"product\s+([a-f0-9]{8,})", txt)
        pid = m.group(1) if m else None
        return {"task": "reviews", "params": {"product_id": pid}}
    if re.search(r"most sold.*(each|per)\s+city|most sold product in each city|most sold product per city", txt):
        return {"task": "most_sold_per_city", "params": {}}
    if re.search(r"most expensive.*(per|in)\s+city|expensive.*per\s+city", txt):
        return {"task": "expensive_per_city", "params": {}}
    if re.search(r"(most expensive|top\s+\d+\s+expensive|top \d+ expensive|list top \d+ expensive)", txt):
        m = re.search(r"top\s+(\d+)", txt)
        topk = int(m.group(1)) if m else 10
        return {"task": "expensive_products", "params": {"topk": topk}}
    if re.search(r"(average order value|aov|average.*price).*in\s+\d{4}", txt):
        y = re.search(r"(\d{4})", txt)
        mcat = re.search(r"for\s+([a-z0-9_ ]+)\s+in", txt)
        cat = mcat.group(1).strip().replace(" ", "_") if mcat else None
        return {"task": "aov", "params": {"year": y.group(1) if y else None, "category": cat}}
    if re.search(r"(what all products|what products are available|what do you sell|what types of products)", txt):
        return {"task": "categories", "params": {}}
    if re.search(r"(top).*cities.*revenue", txt):
        y = re.search(r"(\d{4})", txt)
        return {"task": "top_cities", "params": {"year": y.group(1) if y else None}}
    if re.search(r"(top|most sold|most selling|best sellers).*product", txt):
        m = re.search(r"(\d{1,2})", txt)
        topk = int(m.group(1)) if m else 10
        return {"task": "top_products", "params": {"topk": topk}}
    if re.search(r"(recommend|good for|suggest|best for|what's good)", txt):
        return {"task": "recommend", "params": {}}
    # LLM fallback to parse more complex queries (non-fatal)
    try:
        system = ("Return JSON mapping user request to one of tasks: top_products, expensive_products, expensive_per_city, most_sold_per_city, aov, categories, top_cities, recommend, cheapest, reviews, unknown.")
        prompt = f"User: {user_text}\nReturn valid JSON with 'task' and optional 'params'. Example: {{\"task\":\"top_products\",\"params\":{{\"topk\":10}}}}"
        resp = openai.ChatCompletion.create(
            model=LLM_CLASSIFY_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.replace("```", "")
        parsed = json.loads(raw)
        return parsed
    except Exception:
        return {"task": "unknown", "params": {}}

# -------- recommendations (improved) ----------
def score_candidate(candidate: Dict[str, Any], query_tokens: List[str]) -> float:
    # simple heuristic scoring: higher units_sold and token matches => higher score
    score = 0.0
    units = candidate.get("units_sold") or 0
    avg_price = candidate.get("avg_price") or candidate.get("price") or 0
    score += min(units / 50.0, 5.0)  # scale
    # bonus for token match in category or name
    name = (candidate.get("product_name") or candidate.get("product_category_name") or "").lower()
    for t in query_tokens:
        if t in name:
            score += 1.5
    # prefer mid-priced items slightly (avoid extremely expensive)
    if avg_price and avg_price < 100:
        score += 0.5
    return score

def recommend_products(user_text: str, topk=6):
    # extract keywords from user_text
    tokens = re.findall(r"[a-zA-Z]{3,}", user_text.lower())
    tokens = list(dict.fromkeys(tokens))[:6]
    # candidate pool from categories and fuzzy product names
    candidates = []
    # try fuzzy product search for token combinations
    for t in tokens:
        found = search_products_by_keyword(t, topk=8)
        for f in found:
            if not any(x.get("product_id") == f.get("product_id") for x in candidates):
                candidates.append(f)
        if len(candidates) >= topk * 3:
            break
    # if still small, add popular items
    if len(candidates) < topk:
        popular = top_products_by_units(topk=20)
        for p in popular:
            if not any(x.get("product_id") == p.get("product_id") for x in candidates):
                candidates.append(p)
    # score and pick topk
    scored = []
    for c in candidates:
        s = score_candidate(c, tokens)
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    chosen = [c for s, c in scored[:topk]]
    # enrich with review score if available
    for c in chosen:
        pid = c.get("product_id")
        review_summary = compute_review_score_for_product(pid) if pid else None
        if review_summary:
            c["review_avg"] = review_summary.get("avg_score")
            c["review_n"] = review_summary.get("n")
    return chosen

# -------- top-level agent ----------
def agent_handle(user_text: str, allow_web_search: bool = bool(SERPAPI_KEY)) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    label = classify_intent_openai(user_text)
    txt = user_text.strip()
    evidence = []
    citations = []

    # small talk
    if label == "small_talk" or re.search(r"^\s*(hi|hello|hey|hallo|hiya)\b", txt, re.I):
        reply = "Hey! 👋 I'm your Olist assistant. Ask about products, top sellers, or recommendations (e.g., \"what's good for fiber\")."
        memory_add_conversation(user_text, reply)
        return reply, [], []
    if re.search(r"how are you|how are", txt, re.I):
        reply = "I'm ready — hungry for datasets and helpful answers. What would you like to explore?"
        memory_add_conversation(user_text, reply)
        return reply, [], []

    # explicit translate cue (explain only)
    if re.match(r"^\s*translate\s*$", txt, re.I):
        reply = "Press the Translate button — I will translate only category-like tokens present in the chat (not everything)."
        memory_add_conversation(user_text, reply)
        return reply, [], []

    # parse dataset-style intents and route
    if label in ("data_query", "recommend", "unknown", "translate"):
        parsed = extract_dataset_intent(user_text)
        task = parsed.get("task", "unknown")
        params = parsed.get("params", {}) or {}

        # categories
        if task == "categories":
            rows = list_categories(top_n=60)
            short = "Categories (top shown):\n" + ", ".join([r["product_category_name"] for r in rows[:40]])
            evidence = rows
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # AOV
        if task == "aov":
            year = params.get("year") or (re.search(r"(\d{4})", user_text) and re.search(r"(\d{4})", user_text).group(1))
            category = params.get("category") or (re.search(r"for\s+([a-z0-9_ ]+)\s+in", user_text, re.I) and re.search(r"for\s+([a-z0-9_ ]+)\s+in", user_text, re.I).group(1).strip().replace(" ", "_"))
            if year and category:
                ag = aov_for_category_year(category, year)
                short = f"Average item price for category '{category}' in {year}: {ag.get('avg_price', 0):.2f} (n={ag.get('n', 0)})."
                evidence = [ag]
                reply = format_with_llm(user_text, short, evidence, [])
                memory_add_conversation(user_text, reply)
                return reply, evidence, []
            else:
                reply = "Please include both category and year: e.g. 'Average order value for eletronicos in 2018'."
                memory_add_conversation(user_text, reply)
                return reply, [], []

        # top cities
        if task == "top_cities":
            year = params.get("year") or (re.search(r"(\d{4})", user_text) and re.search(r"(\d{4})", user_text).group(1)) or "2017"
            rows = top_cities_by_revenue(year, topk=5)
            short = "Top cities by revenue:\n" + "\n".join([f"{i+1}. {r['city']} — revenue={r['revenue']:.2f} orders={r['orders']}" for i, r in enumerate(rows)])
            evidence = rows
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # top products
        if task == "top_products":
            topk = params.get("topk", 10)
            year = params.get("year")
            rows = top_products_by_units(topk=topk, year=year)
            short = "Top products by units sold:\n" + "\n".join([f"{i+1}. {r['product_id']} — {r['product_category_name']} units={r['units_sold']}" for i, r in enumerate(rows)])
            evidence = rows
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # expensive products
        if task == "expensive_products":
            topk = params.get("topk", 10)
            rows = expensive_products(topk=topk)
            short = "Most expensive products (by max price):\n" + "\n".join([f"{i+1}. {r['product_id']} — {r['product_category_name']} price={r['max_price']:.2f}" for i, r in enumerate(rows)])
            evidence = rows
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # expensive per city
        if task == "expensive_per_city":
            rows = most_expensive_per_city()
            short = "Most expensive product per city (sample):\n" + "\n".join([f"{r['city']}: {r['product_category_name']} (${r['price']:.2f})" for r in rows[:12]])
            evidence = rows
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # most sold per city
        if task == "most_sold_per_city":
            rows = most_sold_product_per_city()
            short = "Most sold product per city (sample):\n" + "\n".join([f"{r.get('city','-')}: {r.get('product_id','-')} ({r.get('product_category_name','-')}) units={r.get('units_sold',0)}" for r in rows[:12]])
            evidence = rows
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # cheapest product
        if task == "cheapest":
            city = params.get("city")
            rows = cheapest_product(topk=1, city=city)
            if rows:
                r = rows[0]
                short = f"Cheapest product{' in ' + city if city else ''}: {r.get('product_id')} — {r.get('product_category_name')} price={r.get('min_price'):.2f}"
                evidence = rows
                reply = format_with_llm(user_text, short, evidence, [])
                memory_add_conversation(user_text, reply)
                return reply, evidence, []
            else:
                reply = "I couldn't find price data for cheapest products. Try a different query."
                memory_add_conversation(user_text, reply)
                return reply, [], []

        # reviews
        if task == "reviews":
            pid = params.get("product_id")
            # if user asked for "most sold product" reviews, map to top product
            if not pid and re.search(r"most sold", user_text, re.I):
                top = top_products_by_units(topk=1)
                pid = top[0].get("product_id") if top else None
            if not pid:
                reply = "Which product do you want reviews for? Provide a product id or ask 'reviews for the most sold product'."
                memory_add_conversation(user_text, reply)
                return reply, [], []
            reviews = get_reviews_for_product(pid, limit=20)
            score = compute_review_score_for_product(pid)
            if reviews:
                short = f"Found {len(reviews)} reviews for product {pid}."
                if score:
                    short += f" Average score: {score['avg_score']:.2f} (n={score['n']})."
                evidence = reviews[:MAX_EVIDENCE_ROWS]
                reply = format_with_llm(user_text, short, evidence, [])
                memory_add_conversation(user_text, reply)
                return reply, evidence, []
            else:
                reply = f"No customer reviews found in database for product {pid}."
                memory_add_conversation(user_text, reply)
                return reply, [], []

        # recommend
        if task == "recommend" or label == "recommend":
            candidates = recommend_products(user_text, topk=6)
            if not candidates:
                # web fallback optional
                if allow_web_search and SERPAPI_KEY:
                    citations = serpapi_search(user_text, num=3)
                    reply = format_with_llm(user_text, "I searched the web for recommendations.", [], citations)
                    memory_add_conversation(user_text, reply)
                    return reply, [], citations
                reply = "I couldn't find recommendations in the dataset. Try asking differently."
                memory_add_conversation(user_text, reply)
                return reply, [], []
            short = "Recommended dataset-backed products:\n" + "\n".join([f"{i+1}. {p.get('product_id','-')} — {p.get('product_category_name','-')} units={p.get('units_sold',0)}" for i,p in enumerate(candidates)])
            evidence = candidates
            reply = format_with_llm(user_text, short, evidence, [])
            memory_add_conversation(user_text, reply)
            return reply, evidence, []

        # fallback
        reply = "I can answer AOV, top products, cheapest items, per-city queries, reviews, and recommendations. Try: 'cheapest product in Sao Paulo' or 'reviews for <product_id>'."
        memory_add_conversation(user_text, reply)
        return reply, [], []

    # web_search path
    if label == "web_search" or any(w in txt.lower() for w in ("who is", "what is", "when did", "latest", "news")):
        if SERPAPI_KEY:
            citations = serpapi_search(user_text, num=3)
            if citations:
                reply = format_with_llm(user_text, "I searched the web for your question.", [], citations)
                memory_add_conversation(user_text, reply)
                return reply, [], citations
        reply = "I couldn't search the web (SERPAPI_API_KEY not set). I can still answer dataset queries or recommendations."
        memory_add_conversation(user_text, reply)
        return reply, [], []

    # unknown fallback
    reply = "I didn't fully understand. I can chat casually, answer dataset queries (AOV/top sellers/cities/cheapest/reviews), or recommend products. Try one of those."
    memory_add_conversation(user_text, reply)
    return reply, [], []

# -------- UI helpers ----------
def evidence_html_from_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<small>No evidence</small>"
    keys = list(rows[0].keys())
    html = "<div style='font-family:system-ui,Segoe UI,Arial; padding:6px'>"
    html += "<table style='width:100%; border-collapse:collapse; font-size:13px'>"
    html += "<thead><tr>"
    for k in keys:
        html += f"<th style='text-align:left; padding:6px; border-bottom:1px solid #ddd'>{k}</th>"
    html += "</tr></thead><tbody>"
    for r in rows[:MAX_EVIDENCE_ROWS]:
        html += "<tr>"
        for k in keys:
            v = r.get(k, "")
            html += f"<td style='padding:6px; border-bottom:1px solid #f3f3f3'>{v}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

# -------- Gradio UI ----------
CSS = """
.gradio-container { max-width: 1100px; margin: 18px auto; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto; }
#chat_panel { border-radius: 12px; overflow:hidden; height:540px; }
#send_btn button { background:#0b69ff; color:white; border-radius:10px; padding:10px 18px; }
#translate_btn button, #clear_btn button { background:#fff; border:1px solid #ddd; padding:8px 14px; border-radius:10px; }
.msg.user .bubble { background:#DCF8C6; color:#042028; }
.msg.bot .bubble { background:#0b69ff; color:white; }
"""

def gradio_handler(chat_history: Optional[List[Dict[str,Any]]], user_message: str):
    if chat_history is None:
        chat_history = []
    user_message = (user_message or "").strip()
    if not user_message:
        return gr.update(), "", gr.update(value="<small>No evidence</small>"), gr.update(value=None)
    # Append user message in messages format
    chat_history = chat_history + [{"role": "user", "content": user_message}]
    try:
        reply, evidence_rows, citations = agent_handle(user_message, allow_web_search=bool(SERPAPI_KEY))
    except Exception as e:
        tb = traceback.format_exc()
        reply = f"Error: {e}"
        evidence_rows = []
        citations = []
    chat_history = chat_history + [{"role": "assistant", "content": reply}]
    evid_html = evidence_html_from_rows(evidence_rows) if evidence_rows else "<small>No evidence</small>"
    # translations panel will not auto-populate
    return chat_history, "", gr.update(value=evid_html), gr.update(value=None)

def translate_button_click(chat_history: Optional[List[Dict[str,Any]]]):
    """
    Translate only category-like tokens found inside chat messages.
    - Scans messages for tokens containing underscores or known map keys.
    - Does NOT persist translations and does NOT translate the entire translation DB.
    """
    if not chat_history:
        return "<small>No chat content to translate. Ask something first.</small>"
    tokens = set()
    for m in chat_history:
        text = (m.get("content") or "")
        # find tokens with underscores (likely category tokens) and also explicit token mentions
        found = re.findall(r"\b[a-z0-9_]{3,40}\b", text, flags=re.I)
        for t in found:
            if "_" in t and any(c.isalpha() for c in t):
                tokens.add(t.strip())
        # also pick any exact AUTO_TRANSLATIONS keys present
        for k in AUTO_TRANSLATIONS.keys():
            if re.search(rf"\b{k}\b", text):
                tokens.add(k)
    if not tokens:
        return "<small>No category-like tokens found in chat to translate.</small>"
    translations = {}
    for t in sorted(tokens):
        translations[t] = AUTO_TRANSLATIONS.get(t, t.replace("_", " "))
    lines = [f"{k} → {v}" for k, v in translations.items()]
    return "<div style='font-family:system-ui;padding:6px; font-size:14px'>" + "<br>".join(lines) + "</div>"

def clear_chat():
    # keep persistent memory but clear UI chat
    return [], "", gr.update(value="<small>No evidence</small>"), gr.update(value=None)

def build_ui():
    with gr.Blocks(css=CSS, title="Olist Assistant — chat") as demo:
        gr.Markdown("## Olist Assistant — chat\nAsk about products, top sellers, recommendations, or press Translate to convert category tokens to English.")
        with gr.Row():
            with gr.Column(scale=2):
                chat_box = gr.Chatbot(elem_id="chat_panel", label="", type="messages")
                user_input = gr.Textbox(placeholder="Type a message and press Enter — e.g. 'what's good for fiber?'", lines=1)
                with gr.Row():
                    send_btn = gr.Button("Send", elem_id="send_btn")
                    translate_btn = gr.Button("Translate", elem_id="translate_btn")
                    clear_btn = gr.Button("Clear", elem_id="clear_btn")
            with gr.Column(scale=1):
                gr.Markdown("### Evidence / translations")
                evidence_html = gr.HTML("<small>No evidence</small>", elem_id="evidence_html")
                # Do NOT auto-show last translations; show placeholder
                translations_html = gr.HTML("<small>Press Translate to view translations for category tokens in the chat.</small>", elem_id="translations_html")
                gr.Markdown("---")
                gr.Markdown("- Quick tips: Ask casually — typos OK.")
                gr.Markdown("- Examples: 'what all products are available', 'Average order value for eletronicos in 2018', 'most sold product in each city', 'what's good for fiber'")
        # Bind actions
        send_btn.click(fn=gradio_handler, inputs=[chat_box, user_input], outputs=[chat_box, user_input, evidence_html, translations_html])
        user_input.submit(fn=gradio_handler, inputs=[chat_box, user_input], outputs=[chat_box, user_input, evidence_html, translations_html])
        translate_btn.click(fn=translate_button_click, inputs=[chat_box], outputs=[translations_html])
        clear_btn.click(fn=clear_chat, inputs=None, outputs=[chat_box, user_input, evidence_html, translations_html])
    return demo

# -------- main ----------
def main():
    print("Starting quick checks; DB_PATH:", DB_PATH)
    try:
        test = "hi"
        r, ev, cit = agent_handle(test, allow_web_search=False)
        print("QUERY:", test, "->", r[:140], "evidence_rows:", len(ev))
    except Exception as e:
        print("Agent ping failed:", e)
    demo = build_ui()
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT, share=False)

if __name__ == "__main__":
    main()
