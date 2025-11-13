# tests/test_serpapi_full.py
"""
Robust SerpAPI Validation Suite for Olist Assistant

This script ensures the project root is on sys.path so `import src.app` works
regardless of the current working directory or how python is invoked.
"""

import os
import sys
import time
import json

# --- ensure project root is importable (so "import src" works) ---
HERE = os.path.dirname(os.path.abspath(__file__))        # .../tests
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))  # project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# now safe to import from src
try:
    from src.app import serpapi_search
except Exception as e:
    print("ERROR: Failed to import serpapi_search from src.app.")
    print("Reason:", repr(e))
    print("Make sure you're running this from the repository root or that src/ is present.")
    raise

# ------------------------------
# Utility helpers
# ------------------------------
def print_step(msg):
    print(f"\n=== {msg} ===")

def validate_result_structure(result):
    return (
        isinstance(result, dict)
        and "title" in result
        and "link" in result
        and isinstance(result.get("title", ""), str)
        and isinstance(result.get("link", ""), str)
    )

# ------------------------------
# Main test function
# ------------------------------
def run_serpapi_tests():
    print("\n\n====== SerpAPI Extended Test Suite ======\n")

    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        print("⚠️  SERPAPI_API_KEY is not set in environment. Set it to run live SerpAPI tests.")
        print("PowerShell example: $env:SERPAPI_API_KEY = 'your_key_here'")
        print("Exiting test (dry-run mode).")
        return

    queries = [
        ("olist marketplace products", "ecommerce"),
        ("best fiber rich foods", "nutrition"),
        ("best budget smartphones 2024", "electronics"),
        ("top brazil e-commerce platforms", "marketplaces"),
        ("latest openai updates", "news"),
        ("what is olist", "company_lookup"),
        ("furniture decor trends 2024", "dataset-adjacent"),
        ("how to choose a vacuum cleaner", "consumer_advice"),
    ]

    results_summary = {}

    for query, qtype in queries:
        print_step(f"Query: {query}  |  Type: {qtype}")

        try:
            res = serpapi_search(query, num=3)
        except Exception as e:
            print(f"❌ ERROR during request: {e}")
            results_summary[query] = "ERROR"
            continue

        if not res:
            print("❌ FAIL — Empty result list")
            results_summary[query] = "FAIL (empty)"
            continue

        structured = all(validate_result_structure(r) for r in res)
        if not structured:
            print("❌ FAIL — Invalid structure")
            print("Raw:", json.dumps(res, indent=2, ensure_ascii=False))
            results_summary[query] = "FAIL (bad structure)"
            continue

        print("✅ PASS — Valid results returned:")
        for r in res:
            title = r.get("title", "")[:120]
            link = r.get("link", "")
            print(f" - {title}  ({link})")

        results_summary[query] = "PASS"
        time.sleep(1.2)  # polite delay

    # ------------- Summary -------------
    print("\n====== Summary ======\n")
    for q, outcome in results_summary.items():
        print(f"- {q[:40]:40} : {outcome}")
    print("\nDone.\n")

if __name__ == "__main__":
    run_serpapi_tests()
