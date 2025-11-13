# src/db_utils.py
import sqlite3
import pandas as pd
from config import DB_PATH

def run_sql(sql, params=None):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn, params=params or ())
    conn.close()
    return df
