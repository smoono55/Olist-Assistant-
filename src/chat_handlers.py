# src/chat_handlers.py
import re, sqlite3, html
from rapidfuzz import process, fuzz
import pandas as pd
from config import DB_PATH
from db_utils import run_sql

# English->dataset category quick map (extend as needed)
EN_TO_PT_CAT = {
    "electronics": "eletronicos",
    "electronic": "eletronicos",
    "eletrodomesticos": "eletrodomesticos",
    "home": "casa_conforto",
    "food": "alimentos",
    "drink": "bebidas",
    "beverages": "bebidas",
    "toys": "brinquedos",
    "furniture": "moveis_decoracao",
    "phone": "telefonia",
    "phones": "telefonia",
    "veggie": "alimentos",
    "vegetable": "alimentos",
    "fiber": "alimentos"
}

def get_all_categories():
    sql = "SELECT DISTINCT product_category_name FROM orders_items WHERE product_category_name IS NOT NULL"
    df = run_sql(sql)
    return [c for c in df['product_category_name'].tolist() if c and str(c).strip()!='nan']

def fuzzy_match_category(term):
    if not term:
        return None
    t = term.lower().strip()
    if t in EN_TO_PT_CAT:
        return EN_TO_PT_CAT[t]
    cats = get_all_categories()
    best = process.extractOne(t, cats, scorer=fuzz.WRatio)
    if best and best[1] >= 75:
        return best[0]
    # fallback: replace spaces with underscores
    return t.replace(" ", "_")

def analytics_direct_handler(query):
    q = (query or "").lower().strip()

    # AOV pattern (flexible)
    m = re.search(r"(average|avg|mean).*(order value|aov).*for\s+([a-z0-9_\- ]+?)\s+in\s+(\d{4})", q)
    if m:
        raw_cat = m.group(3).strip()
        cat = fuzzy_match_category(raw_cat)
        year = m.group(4)
        sql = "SELECT AVG(price) as avg_price FROM orders_items WHERE LOWER(product_category_name)=? AND STRFTIME('%Y', order_purchase_timestamp)=?"
        df = run_sql(sql, params=(cat, year))
        val = float(df.iloc[0,0]) if df.shape[0]>0 and df.iloc[0,0] is not None else None
        return {"matched": True, "type":"aov", "category": cat, "year": year, "value": val, "sql": sql, "params": (cat, year)}

    # Top N cities by revenue
    m = re.search(r"(top|op|highest|largest)\s*(\d+)?\s*(cities|city).*?(revenue|sales).*?(\d{4})", q)
    if m:
        n = int(m.group(2) or 5)
        year = m.group(5)
        sql = """SELECT customer_city AS city, SUM(price) AS revenue
                 FROM orders_items
                 WHERE STRFTIME('%Y', order_purchase_timestamp)=?
                 GROUP BY customer_city
                 ORDER BY revenue DESC
                 LIMIT ?"""
        df = run_sql(sql, params=(year, n))
        return {"matched": True, "type":"top_cities", "year": year, "df": df, "sql": sql, "params": (year, n)}

    # list categories
    if any(phrase in q for phrase in ["what all products", "what products", "what categories", "products available", "what all products are available", "what all products"]):
        cats = get_all_categories()
        # include counts sample
        sql = "SELECT product_category_name, COUNT(*) as cnt FROM orders_items GROUP BY product_category_name ORDER BY cnt DESC LIMIT 100"
        df = run_sql(sql)
        return {"matched": True, "type":"list_categories", "df": df}

    return {"matched": False}
