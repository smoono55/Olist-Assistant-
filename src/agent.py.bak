# src/agent.py
"""
Compatibility / agent shim for the Olist Assistant.

This module exports `agent_handle(user_text: str) -> (reply:str, rows:list[dict], html:str)`.

It is intentionally self-contained and uses the local SQLite DB created by your ETL (data/olist_analytics.db).
Purpose:
 - Fix import errors (uvicorn startup requires agent_handle)
 - Provide a robust baseline agent that can be iterated later (RAG, OpenAI, web search can be added)
"""

from __future__ import annotations
import os
import sqlite3
import re
import html as htmllib
from typing import List, Dict, Tuple, Any

# --- Configuration: DB location (same as other modules expect) ---
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "olist_analytics.db"))

# Ensure DB exists (do not crash import if missing; we will return friendly errors at runtime)
_DB_EXISTS = os.path.exists(DB_PATH)

# --- DB helpers ---
def _db_connect():
    if not _DB_EXISTS:
        raise RuntimeError(f"Database not found at {DB_PATH}. Run the ETL first.")
    return sqlite3.connect(DB_PATH)

def run_sql(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """
    Run a read-only SQL query and return list of rows as dicts.
    Designed for simple SELECT aggregation queries used by the agent.
    """
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return rows
    finally:
        conn.close()

# --- Small utilities ---
def _rows_to_html(rows: List[Dict[str, Any]], limit: int = 8) -> str:
    if not rows:
        return "<small>No evidence rows.</small>"
    keys = list(rows[0].keys())
    html = "<div style='font-family:system-ui,Segoe UI,Arial; max-width:600px;'>"
    html += "<table style='width:100%; border-collapse:collapse; font-size:13px'>"
    html += "<thead><tr>"
    for k in keys:
        html += f"<th style='text-align:left; padding:6px; border-bottom:1px solid #eee'>{htmllib.escape(str(k))}</th>"
    html += "</tr></thead><tbody>"
    for r in rows[:limit]:
        html += "<tr>"
        for k in keys:
            v = "" if r.get(k) is None else htmllib.escape(str(r.get(k)))
            html += f"<td style='padding:6px; border-bottom:1px solid #fafafa'>{v}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

def _short_reply_template(lines: List[str]) -> str:
    # produce a short casual reply (2-4 lines)
    return " ".join(lines)

# --- Basic "intent" detection (lightweight, local) ---
def _classify_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("hi", "hello", "hey", "yo", "hii", "hiya")):
        return "small_talk"
    if any(w in t for w in ("recommend", "what's good", "what is good", "good for", "suggest", "best for", "help with")):
        return "recommend"
    if any(w in t for w in ("average", "aov", "order value", "top", "most selling", "most sold", "list", "available", "what all", "products", "cities", "revenue")):
        return "data_query"
    if any(w in t for w in ("translate", "what language", "english", "portuguese", "mean")):
        return "translate"
    # fallback
    return "unknown"

# --- Recommendation fallback: pick top products from likely food categories for nutrient queries ---
NUTRITION_CATEGORIES = {
    "fiber": ["alimentos", "alimentos_bebidas", "bebidas"],
    "gut": ["alimentos", "alimentos_bebidas"],
    "eyesight": ["alimentos", "alimentos_bebidas", "perfumaria"],  # coarse fallback
}

def recommend_for_keyword(keyword: str, top_k: int = 6) -> Tuple[str, List[Dict[str,Any]]]:
    # map keyword to nutrition-ish categories
    key = keyword.lower()
    categories = []
    if "fiber" in key or "fibra" in key or "fibre" in key:
        categories = NUTRITION_CATEGORIES["fiber"]
    elif "gut" in key or "gut" in key or "stomach" in key or "gut" in key:
        categories = NUTRITION_CATEGORIES["gut"]
    elif "eye" in key or "vision" in key or "sight" in key:
        categories = NUTRITION_CATEGORIES.get("eyesight", [])
    else:
        categories = ["alimentos", "alimentos_bebidas"]

    placeholders = ",".join("?" for _ in categories)
    sql = f"""
        SELECT product_id, product_category_name, COUNT(*) as units_sold, AVG(price) as avg_price
        FROM orders_items
        WHERE product_category_name IN ({placeholders})
        GROUP BY product_id
        ORDER BY units_sold DESC
        LIMIT ?
    """
    rows = run_sql(sql, tuple(categories) + (top_k,))
    if not rows:
        return ("No matching products found in food categories.", [])
    # build human summary
    lines = [f"Found {len(rows)} candidate product(s) for {keyword} (from dataset):"]
    for i, r in enumerate(rows[:top_k]):
        lines.append(f"{i+1}. {r.get('product_id')} — {r.get('product_category_name')} units_sold={r.get('units_sold')}")
    return (" ".join(lines), rows)

# --- Data query handlers ---
def handle_list_categories() -> Tuple[str, List[Dict[str,Any]]]:
    rows = run_sql("""
        SELECT product_category_name, COUNT(*) as cnt
        FROM orders_items
        GROUP BY product_category_name
        ORDER BY cnt DESC
        LIMIT 60
    """)
    if not rows:
        return ("No categories found in dataset.", [])
    # brief reply
    top = ", ".join([r['product_category_name'] or "None" for r in rows[:20]])
    reply = f"Categories (top shown): {top}"
    return (reply, rows)

def handle_top_products(limit: int = 10, year: str|None = None) -> Tuple[str, List[Dict[str,Any]]]:
    if year:
        rows = run_sql("""
            SELECT product_id, product_category_name, COUNT(*) as units_sold, SUM(item_total) as revenue
            FROM orders_items
            WHERE STRFTIME('%Y', order_purchase_timestamp) = ?
            GROUP BY product_id
            ORDER BY units_sold DESC
            LIMIT ?
        """, (year, limit))
    else:
        rows = run_sql("""
            SELECT product_id, product_category_name, COUNT(*) as units_sold, SUM(item_total) as revenue
            FROM orders_items
            GROUP BY product_id
            ORDER BY units_sold DESC
            LIMIT ?
        """, (limit,))
    if not rows:
        return ("No product sales found.", [])
    reply_lines = [f"Top products by units sold:"]
    for i, r in enumerate(rows[:limit]):
        reply_lines.append(f"{i+1}. {r['product_id']} — {r['product_category_name']} units={r['units_sold']}")
    return (" ".join(reply_lines), rows)

def handle_aov(category: str, year: str) -> Tuple[str, List[Dict[str,Any]]]:
    rows = run_sql("""
        SELECT AVG(price) as avg_price, COUNT(*) as n
        FROM orders_items
        WHERE product_category_name = ? AND STRFTIME('%Y', order_purchase_timestamp) = ?
    """, (category, year))
    if not rows:
        return (f"No data for {category} in {year}.", [])
    ag = rows[0]
    avg = ag.get('avg_price') or 0
    n = ag.get('n') or 0
    reply = f"Average item price for category '{category}' in {year}: {avg:.2f} (n={n})."
    return (reply, [ag])

def handle_top_cities(year: str, topk: int = 5) -> Tuple[str, List[Dict[str,Any]]]:
    rows = run_sql("""
        SELECT customer_city as city, SUM(item_total) as revenue, COUNT(*) as orders
        FROM orders_items
        WHERE STRFTIME('%Y', order_purchase_timestamp)=?
        GROUP BY customer_city
        ORDER BY revenue DESC
        LIMIT ?
    """, (year, topk))
    if not rows:
        return (f"No city revenue data for {year}.", [])
    lines = [f"Top {len(rows)} cities by revenue in {year}:"]
    for i,r in enumerate(rows):
        lines.append(f"{i+1}. {r['city']} — revenue={r['revenue']:.2f} orders={r['orders']}")
    return (" ".join(lines), rows)

# --- A tiny translator for category tokens (Portuguese -> short English) ---
_CATEGORY_TRANSLATIONS = {
    "moveis_decoracao": "furniture & decor",
    "beleza_saude": "beauty & health",
    "cama_mesa_banho": "bed, bath & linen",
    "relogios_presentes": "watches & gifts",
    "informatica_acessorios": "computer accessories",
    "ferramentas_jardim": "tools & garden",
    "eletronicos": "electronics",
    "bebes": "baby products",
    "papelaria": "stationery",
    "perfumaria": "perfumery",
    "alimentos": "food",
    "alimentos_bebidas": "food & drinks",
    # add more as needed
}

def translate_categories(rows: List[Dict[str,Any]]) -> Dict[str,str]:
    out = {}
    for r in rows:
        key = r.get("product_category_name")
        if key is None:
            continue
        out[key] = _CATEGORY_TRANSLATIONS.get(key, key.replace("_", " "))
    return out

# --- Public entrypoint used by ASGI wrapper ---
def agent_handle(user_text: str) -> Tuple[str, List[Dict[str,Any]], str]:
    """
    Main function expected by src.asgi. Returns (reply, evidence_rows, evidence_html).
    Keep responses concise and friendly.
    """
    if not user_text:
        return ("Say something and I'll help — ask about products, sales, or recommendations.", [], "")
    # if DB missing, return error (do not throw on import)
    if not _DB_EXISTS:
        return (f"Local DB not found at {DB_PATH}. Please run the ETL and imports first.", [], "")

    text = user_text.strip()
    intent = _classify_intent(text)

    try:
        if intent == "small_talk":
            t = text.lower()
            if any(w in t for w in ("hi", "hello", "hey")):
                return ("Hey! 👋 I'm your Olist assistant. Ask about products, top sellers, or recommendations (e.g., 'what's good for fiber').", [], "")
            if "how are" in t:
                return ("I'm a dataset-hungry bot — feeling ready! What can I do for you today?", [], "")
            return ("Nice to chat — I can answer product and sales questions. Try 'what all products are available' or 'Top 5 cities by revenue in 2017'.", [], "")

        if intent == "recommend":
            # try to detect nutrient/need
            if re.search(r"fiber|fibra|fibre", text, re.I):
                summary, rows = recommend_for_keyword("fiber", top_k=6)
                html = _rows_to_html(rows)
                return (summary, rows, html)
            if re.search(r"gut|stomach|digestion|gut health", text, re.I):
                summary, rows = recommend_for_keyword("gut", top_k=6)
                html = _rows_to_html(rows)
                return (summary, rows, html)
            if re.search(r"eye|vision|sight", text, re.I):
                summary, rows = recommend_for_keyword("eyesight", top_k=6)
                html = _rows_to_html(rows)
                return (summary, rows, html)
            # generic recommend fallback
            summary, rows = recommend_for_keyword(text, top_k=6)
            html = _rows_to_html(rows)
            return (summary, rows, html)

        if intent == "translate":
            # if user asks translate recent categories, we cannot access session here,
            # but we accept input like "translate cama_mesa_banho, beleza_saude"
            cats = re.findall(r"[a-z_]+", text.lower())
            if not cats:
                return ("Please provide categories to translate (e.g. 'translate cama_mesa_banho').", [], "")
            mapping = {c: _CATEGORY_TRANSLATIONS.get(c, c.replace("_"," ")) for c in cats}
            lines = [f"{k} → {v}" for k,v in mapping.items()]
            return ("Translations:\n" + "\n".join(lines), [{"product_category_name": k} for k in mapping.keys()], _rows_to_html([{"product_category_name": k, "translation": v} for k,v in mapping.items()]))

        if intent == "data_query":
            # AOV pattern
            m = re.search(r"average.*(?:order value|aov|price).*for\s+([a-z0-9_ ]+)\s+in\s+(\d{4})", text, re.I)
            if m:
                category = m.group(1).strip().replace(" ", "_")
                year = m.group(2)
                reply, rows = handle_aov(category, year)
                return (reply, rows, _rows_to_html(rows))

            # top cities by revenue
            if re.search(r"top .*cities.*revenue|top (\d+) cities", text, re.I):
                year_m = re.search(r"(\d{4})", text)
                year = year_m.group(1) if year_m else "2017"
                topk_m = re.search(r"top\s+(\d+)", text, re.I)
                topk = int(topk_m.group(1)) if topk_m else 5
                reply, rows = handle_top_cities(year, topk)
                return (reply, rows, _rows_to_html(rows))

            # top selling products
            if re.search(r"top .*selling|most selling|most sold|top .* products|best sellers|most sold items", text, re.I):
                # optional year
                year_m = re.search(r"in\s+(\d{4})", text)
                year = year_m.group(1) if year_m else None
                reply, rows = handle_top_products(limit=10, year=year)
                return (reply, rows, _rows_to_html(rows))

            # list categories / products available
            if re.search(r"what all products|what(?:'s| is) available|what products|list products|what do u sell|what all do u sell", text, re.I):
                reply, rows = handle_list_categories()
                return (reply, rows, _rows_to_html(rows))

            # fallback: try to find category mentioned + year
            m2 = re.search(r"for\s+([a-z_ ]+)\s+in\s+(\d{4})", text, re.I)
            if m2:
                cat = m2.group(1).strip().replace(" ", "_")
                yr = m2.group(2)
                return handle_aov(cat, yr) + ("",)

            # final fallback for data queries
            return ("I can answer AOV, top products, top cities, and list categories. Examples: 'Average order value for eletronicos in 2018' or 'Top 5 cities by revenue in 2017'.", [], "")

        # unknown: attempt a helpful fallback: if user contains a category token, run a short sample
        token = re.search(r"\b([a-z_]{3,40})\b", text.lower())
        if token:
            cand = token.group(1)
            # quick check if that token looks like category
            rows = run_sql("SELECT product_id, product_category_name, COUNT(*) as units_sold FROM orders_items WHERE product_category_name = ? GROUP BY product_id ORDER BY units_sold DESC LIMIT 5", (cand,))
            if rows:
                reply = f"Showing top items for category '{cand}': " + ", ".join([r['product_id'] for r in rows[:5]])
                return (reply, rows, _rows_to_html(rows))
        # total fallback
        return ("I didn't fully understand. I can chat casually, answer dataset queries (AOV/top sellers/cities), or recommend products (e.g., 'what's good for fiber'). Try one of those.", [], "")
    except Exception as ex:
        # do not crash — return friendly error message and empty evidence
        return (f"Agent runtime error: {ex}", [], "")

# Expose name for import checks
__all__ = ["agent_handle"]
