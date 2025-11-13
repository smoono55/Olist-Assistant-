# scripts/import_products_to_sqlite.py
"""
Import olist_products_dataset.csv into the existing olist_analytics.db SQLite database.

Usage:
    python scripts/import_products_to_sqlite.py --csv data/olist_products_dataset.csv --db data/olist_analytics.db
"""

import argparse
import os
import sqlite3
import pandas as pd

def main(csv_path, db_path):
    print(f"[import] CSV: {csv_path}")
    print(f"[import] DB:  {db_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    # load CSV
    df = pd.read_csv(csv_path)
    print(f"[import] read {len(df):,} rows")

    df.columns = [c.strip() for c in df.columns]
    if "product_id" not in df.columns:
        raise RuntimeError("CSV missing required column: product_id")

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("olist_products", conn, if_exists="replace", index=False)
        print("[import] wrote table: olist_products")
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS ix_products_product_id ON olist_products(product_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_products_category ON olist_products(product_category_name);")
        conn.commit()
        print("[import] created indexes.")
    finally:
        conn.close()

    print("[import] DONE. You can now restart the app and query products.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/olist_products_dataset.csv", help="Path to CSV file")
    parser.add_argument("--db", default="data/olist_analytics.db", help="Path to SQLite DB")
    args = parser.parse_args()
    main(args.csv, args.db)
