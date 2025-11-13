# src/etl.py
"""
Robust ETL for the Olist dataset.

- Looks for CSV files in data/ relative to project root, or in project root if missing.
- Builds a denormalized table: order_items joined with orders, customers, products, payments, reviews, geolocation.
- Normalizes timestamps and numeric fields.
- Writes: data/canonical_orders.parquet and data/olist_analytics.db (table orders_items).
"""
from pathlib import Path
import pandas as pd
import sqlite3
import sys
import traceback

# Try to import config if present (preferred)
try:
    from config import DATA_DIR, CANONICAL_PARQUET, DB_PATH
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    CANONICAL_PARQUET = DATA_DIR / "canonical_orders.parquet"
    DB_PATH = DATA_DIR / "olist_analytics.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# CSV filenames we expect (your list)
CSV_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_trans": "product_category_name_translation.csv",
}

def locate_csv(name):
    """Return Path for a given CSV name: prefer data/ but also check project root."""
    p1 = DATA_DIR / name
    if p1.exists():
        return p1
    alt = Path.cwd() / name
    if alt.exists():
        return alt
    return None

def read_csv_safe(p: Path, **kwargs):
    if p is None:
        print(f"[etl] missing: {p} (None)")
        return pd.DataFrame()
    try:
        print(f"[etl] loading {p.name} ...")
        return pd.read_csv(p, low_memory=False, **kwargs)
    except Exception as e:
        print(f"[etl] failed to read {p}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def coerce_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def safe_numeric(df, col, fill=0.0):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)
    else:
        df[col] = fill
    return df

def build_canonical():
    # Locate files
    paths = {k: locate_csv(v) for k, v in CSV_FILES.items()}
    for k,v in paths.items():
        if v is None:
            print(f"[etl] warning: {CSV_FILES[k]} not found in data/ or project root.")

    # Load
    customers = read_csv_safe(paths["customers"])
    geoloc = read_csv_safe(paths["geolocation"])
    items = read_csv_safe(paths["order_items"])
    payments = read_csv_safe(paths["order_payments"])
    reviews = read_csv_safe(paths["order_reviews"])
    orders = read_csv_safe(paths["orders"])
    products = read_csv_safe(paths["products"])
    sellers = read_csv_safe(paths["sellers"])
    cat_trans = read_csv_safe(paths["category_trans"])

    # Basic validation
    if items.empty:
        raise RuntimeError("[etl] order_items CSV is required but missing or empty. Place it in data/ and retry.")

    # Merge items <- orders
    print("[etl] merging items -> orders ...")
    df = items.copy()
    if not orders.empty:
        df = df.merge(orders, on="order_id", how="left", suffixes=("", "_orders"))

    # Merge customers
    if not customers.empty and "customer_id" in df.columns:
        print("[etl] merging customers ...")
        df = df.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))

    # Merge products
    if not products.empty and "product_id" in df.columns:
        print("[etl] merging products ...")
        df = df.merge(products, on="product_id", how="left", suffixes=("", "_prod"))

    # Merge payments (aggregate payment_value per order if multiple rows)
    if not payments.empty:
        print("[etl] aggregating & merging payments ...")
        try:
            payments_agg = payments.copy()
            # ensure numeric
            payments_agg['payment_value'] = pd.to_numeric(payments_agg.get('payment_value', 0), errors='coerce').fillna(0)
            payments_agg = payments_agg.groupby("order_id", as_index=False).agg({
                "payment_value": "sum",
                "payment_type": lambda s: "|".join(map(str, s.dropna().unique())) if 'payment_type' in payments_agg.columns else ""
            })
            df = df.merge(payments_agg, on="order_id", how="left")
        except Exception:
            print("[etl] payment merge failed, continuing without payments.")
            traceback.print_exc()

    # Merge reviews (keep review_score etc.)
    if not reviews.empty:
        print("[etl] merging reviews ...")
        try:
            reviews_small = reviews[['order_id','review_score','review_creation_date']].copy()
            df = df.merge(reviews_small, on="order_id", how="left")
            # coerce review score numeric
            if 'review_score' in df.columns:
                df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')
        except Exception:
            print("[etl] reviews merge failed, continuing.")
            traceback.print_exc()

    # Merge seller info if present (items have seller_id)
    if not sellers.empty and 'seller_id' in df.columns:
        print("[etl] merging sellers ...")
        try:
            df = df.merge(sellers, on="seller_id", how="left", suffixes=("", "_seller"))
        except Exception:
            print("[etl] sellers merge failed.")
            traceback.print_exc()

    # Optionally join geolocation by zip code prefix if both exist
    # Many datasets use *_zip_code_prefix fields; we'll try a join on customer_zip_code_prefix -> geolocation_zip_code_prefix
    if not geoloc.empty:
        left_col = None
        if 'customer_zip_code_prefix' in df.columns:
            left_col = 'customer_zip_code_prefix'
        elif 'seller_zip_code_prefix' in df.columns:
            left_col = 'seller_zip_code_prefix'
        if left_col and 'geolocation_zip_code_prefix' in geoloc.columns:
            print("[etl] merging geolocation on zip code prefix ...")
            try:
                # coerce to same type for join
                ge = geoloc.rename(columns={
                    'geolocation_zip_code_prefix': 'zip_prefix',
                    'geolocation_lat': 'geo_lat',
                    'geolocation_lng': 'geo_lng',
                    'geolocation_city': 'geo_city',
                    'geolocation_state': 'geo_state'
                })
                ge['zip_prefix'] = ge['zip_prefix'].astype(str)
                df[left_col] = df[left_col].astype(str)
                df = df.merge(ge[['zip_prefix','geo_lat','geo_lng','geo_city','geo_state']].drop_duplicates('zip_prefix'),
                              left_on=left_col, right_on='zip_prefix', how='left')
            except Exception:
                print("[etl] geolocation merge failed.")
                traceback.print_exc()

    # Apply category translation if available (English translations)
    if not cat_trans.empty and 'product_category_name' in df.columns:
        print("[etl] applying product category translation (if available) ...")
        try:
            ct = cat_trans.rename(columns={
                'product_category_name': 'product_category_name',
                'product_category_name_english': 'product_category_name_english'
            })
            df = df.merge(ct, on='product_category_name', how='left')
        except Exception:
            print("[etl] category translation failed.")
            traceback.print_exc()

    # Normalize timestamps & numeric fields
    print("[etl] normalizing timestamps & numeric columns ...")
    ts_cols = ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']
    for c in ts_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    # Derive order year/month
    if 'order_purchase_timestamp' in df.columns:
        df['order_year'] = df['order_purchase_timestamp'].dt.year
        df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    else:
        df['order_year'] = pd.NA
        df['order_month'] = pd.NA

    # Ensure numeric columns
    safe_numeric_cols = ['price','freight_value','item_total']
    for c in safe_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    # if item_total not present, compute it
    if 'item_total' not in df.columns:
        df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0.0)
        df['freight_value'] = pd.to_numeric(df.get('freight_value', 0), errors='coerce').fillna(0.0)
        df['item_total'] = df['price'] + df['freight_value']

    # Column tidy: ensure strings do not break parquet/sqlite
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).where(df[col].notnull(), None)

    # Write canonical parquet and sqlite
    try:
        CANONICAL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        print(f"[etl] writing parquet -> {CANONICAL_PARQUET} ...")
        df.to_parquet(CANONICAL_PARQUET, index=False, engine='auto')
        print(f"[etl] parquet written ({len(df)} rows).")
    except Exception:
        print("[etl] failed to write parquet, attempting fallback to .csv ...")
        traceback.print_exc()
        try:
            csvp = CANONICAL_PARQUET.with_suffix('.csv')
            df.to_csv(csvp, index=False)
            print("[etl] fallback CSV written at", csvp)
        except Exception:
            print("[etl] failed fallback CSV write.")
            traceback.print_exc()

    # Write SQLite DB (orders_items table)
    try:
        if DB_PATH.exists():
            print("[etl] removing existing DB:", DB_PATH)
            DB_PATH.unlink()
        print("[etl] writing sqlite DB ->", DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("orders_items", conn, index=False)
        conn.close()
        print("[etl] sqlite DB written with table 'orders_items'.")
    except Exception:
        print("[etl] failed to write sqlite DB.")
        traceback.print_exc()

    print("[etl] DONE.")

if __name__ == "__main__":
    try:
        build_canonical()
    except Exception as e:
        print("[etl] fatal error:", e)
        traceback.print_exc()
        sys.exit(2)

