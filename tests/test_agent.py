# tests/test_agent.py
"""
Automated test harness for the Olist Agent.

Usage:
    python tests/test_agent.py

This file imports `agent_handle` from src.app and tests how well it
responds to a large set of realistic user queries. It does NOT modify app.py.
"""

import sys
import os
import traceback

# --- Adjust import path ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from app import agent_handle
except Exception as e:
    print("❌ ERROR importing agent_handle from src/app.py:", e)
    print(traceback.format_exc())
    sys.exit(1)

# -------------------------
# TEST CASES
# -------------------------

TEST_MESSAGES = [
    # Small talk
    "hi",
    "hello",
    "how are you",
    "what can you do?",

    # Categories
    "what all products are available",
    "list the different types of products you sell",

    # AOV queries
    "Average order value for eletronicos in 2018",
    "average price for moveis_decoracao in 2017",

    # Top sellers
    "top 5 cities by revenue in 2017",
    "list top 10 most selling products",
    "which are the most sold products?",
    "most sold product per city",
    "most sold product in each city",

    # Price-related
    "what is the cheapest product",
    "cheapest product in sao paulo",
    "list top 5 expensive products",
    "most expensive product per city",

    # Recommendations / nutrition
    "what's good for fiber",
    "give me food good for gut health",
    "suggest something good for eyesight",
    "recommend a product for immunity",

    # Reviews
    "what is the customer review on the most sold product",
    "reviews for product aca2eb7d00ea1a7b8ebd4e68314663af",
    "what is the review score of product 99a4788cb24856965c36a24e339b6058",

    # Web-search-like (should be dataset fallback or short refusal)
    "who is the founder of olist",
    "what is machine learning",
    "latest news about brazil economy",

    # Unknown / strange
    "ajslkdjas aslkdj laskdj",
    "can you fly",
    "tell me something random",
]

# -------------------------
# EVALUATION RULES
# -------------------------

def is_valid_response(text: str) -> bool:
    """Define what counts as a 'good' response."""
    if not text or len(text.strip()) < 3:
        return False
    if "Error:" in text:
        return False
    if "I couldn't search the web" in text and "dataset" not in text:
        return False
    return True


# -------------------------
# TEST RUNNER
# -------------------------

def run_tests():
    print("\n===== OLIST AGENT AUTOMATED TEST SUITE =====\n")

    passed = 0
    failed = 0
    detailed_failures = []

    for msg in TEST_MESSAGES:
        try:
            reply, _, _ = agent_handle(msg, allow_web_search=False)
        except Exception as e:
            failed += 1
            detailed_failures.append((msg, f"EXCEPTION: {e}"))
            continue

        if is_valid_response(reply):
            passed += 1
            print(f"✔ PASS — '{msg}'\n  → {reply[:80]}...\n")
        else:
            failed += 1
            detailed_failures.append((msg, reply))
            print(f"❌ FAIL — '{msg}'\n  → {reply}\n")

    # Summary
    print("\n============================================")
    print("FINAL RESULTS")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print("============================================\n")

    # Print details of failures
    if failed > 0:
        print("❗ FAILURES DETAIL:\n")
        for msg, out in detailed_failures:
            print(f"--- Query: {msg}\nOutput: {out}\n")

    print("Done.\n")


if __name__ == "__main__":
    run_tests()
