"""
================================================================================
Database Validation Tests
================================================================================
Confirms the SQLite database matches the validated 4,109-day x 23-ticker dataset.
Runs 8 checks against known ground truth from the reconciled pickle.
================================================================================
"""
import sqlite3
import pandas as pd
import numpy as np
import os

# Import data access functions
from data_access import get_prices, get_ticker_info, get_calendar, get_close_matrix

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "quant.db")
PICKLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reconciled", "nifty_subset_reconciled.pkl")

EXPECTED_TICKERS = 23
EXPECTED_DAYS = 4109
EXPECTED_TOTAL_ROWS = EXPECTED_TICKERS * EXPECTED_DAYS  # 94,507

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print("  [PASS] %s" % name)
    else:
        failed += 1
        print("  [FAIL] %s" % name)
    if detail:
        print("         %s" % detail)


print("=" * 70)
print("DATABASE VALIDATION TESTS")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# =========================================================================
# Test 1: Total row count in daily_prices
# =========================================================================
print("\n--- Test 1: Total Row Count ---")
cur.execute("SELECT COUNT(*) FROM daily_prices")
total_rows = cur.fetchone()[0]
check("daily_prices has %d rows (expected %d)" % (total_rows, EXPECTED_TOTAL_ROWS),
      total_rows == EXPECTED_TOTAL_ROWS,
      "Got: %d" % total_rows)

# =========================================================================
# Test 2: Ticker count
# =========================================================================
print("\n--- Test 2: Ticker Count ---")
cur.execute("SELECT COUNT(*) FROM tickers")
ticker_count = cur.fetchone()[0]
check("tickers table has %d rows (expected %d)" % (ticker_count, EXPECTED_TICKERS),
      ticker_count == EXPECTED_TICKERS)

# =========================================================================
# Test 3: Per-ticker row counts (each should have 4,109)
# =========================================================================
print("\n--- Test 3: Per-Ticker Row Counts ---")
cur.execute("""
    SELECT t.symbol, COUNT(*) as row_count
    FROM daily_prices dp
    JOIN tickers t ON dp.ticker_id = t.ticker_id
    GROUP BY t.symbol
    ORDER BY t.symbol
""")
per_ticker = cur.fetchall()
all_match = True
for symbol, count in per_ticker:
    if count != EXPECTED_DAYS:
        all_match = False
        print("  [FAIL] %-16s has %d rows (expected %d)" % (symbol, count, EXPECTED_DAYS))
    else:
        print("  [ OK ] %-16s %d rows" % (symbol, count))

check("All tickers have exactly %d rows" % EXPECTED_DAYS, all_match)

# =========================================================================
# Test 4: Trading calendar count
# =========================================================================
print("\n--- Test 4: Trading Calendar ---")
cur.execute("SELECT COUNT(*) FROM trading_calendar")
cal_count = cur.fetchone()[0]
check("trading_calendar has %d rows (expected %d)" % (cal_count, EXPECTED_DAYS),
      cal_count == EXPECTED_DAYS)

# =========================================================================
# Test 5: Date range
# =========================================================================
print("\n--- Test 5: Date Range ---")
cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices")
min_date, max_date = cur.fetchone()
check("First date = 2010-01-04", min_date == "2010-01-04",
      "Got: %s" % min_date)
check("Last date = 2026-08-28", max_date == "2026-08-28",
      "Got: %s" % max_date)

# =========================================================================
# Test 6: Data quality breakdown
# =========================================================================
print("\n--- Test 6: Data Quality Breakdown ---")
cur.execute("""
    SELECT data_quality, COUNT(*) as cnt
    FROM daily_prices
    GROUP BY data_quality
    ORDER BY data_quality
""")
quality_breakdown = cur.fetchall()
print("  Data quality counts:")
for quality, count in quality_breakdown:
    print("    %-20s %d rows" % (quality, count))

total_quality = sum(c for _, c in quality_breakdown)
check("Quality counts sum to total rows",
      total_quality == EXPECTED_TOTAL_ROWS,
      "Sum: %d, Expected: %d" % (total_quality, EXPECTED_TOTAL_ROWS))

# =========================================================================
# Test 7: Spot-check — get_prices for TCS.NS in 2020
# =========================================================================
print("\n--- Test 7: Spot Check — TCS.NS 2020 ---")
tcs = get_prices("TCS.NS", "2020-01-01", "2020-12-31")
print("  TCS.NS 2020 rows: %d" % len(tcs))
print("  Columns: %s" % list(tcs.columns))
print("  Head:")
print(tcs.head().to_string().replace("\n", "\n  "))
check("TCS.NS 2020 query returns data", len(tcs) > 200,
      "Got %d rows" % len(tcs))

# =========================================================================
# Test 8: Cross-validation — DB close prices vs pickle
# =========================================================================
print("\n--- Test 8: Cross-Validation (DB vs Pickle) ---")
rec = pd.read_pickle(PICKLE_PATH)
tickers_in_data = sorted(set(c[0] for c in rec.columns))

max_diff = 0
mismatched_tickers = []

for t in tickers_in_data:
    # Get from DB
    db_prices = get_prices(t, "2010-01-01", "2026-12-31")
    # Get from pickle
    pkl_close = rec[t]["Close"]

    # Compare close prices
    common_dates = db_prices.index.intersection(pkl_close.index)
    if len(common_dates) == 0:
        mismatched_tickers.append((t, "no common dates"))
        continue

    db_close = db_prices.loc[common_dates, "close"].values
    pkl_vals = pkl_close.loc[common_dates].values

    diff = np.nanmax(np.abs(db_close - pkl_vals))
    max_diff = max(max_diff, diff)

    if diff > 0.01:
        mismatched_tickers.append((t, "max diff: %.6f" % diff))

check("DB close prices match pickle (max diff < 0.01)",
      max_diff < 0.01 and len(mismatched_tickers) == 0,
      "Max diff across all tickers: %.6f" % max_diff)

if mismatched_tickers:
    for t, detail in mismatched_tickers:
        print("    [MISMATCH] %s: %s" % (t, detail))

# =========================================================================
# Test 9: Sector distribution
# =========================================================================
print("\n--- Test 9: Sector Distribution ---")
cur.execute("""
    SELECT sector, COUNT(*) as cnt
    FROM tickers
    GROUP BY sector
    ORDER BY sector
""")
sectors = cur.fetchall()
print("  Sector breakdown:")
for sector, count in sectors:
    print("    %-35s %d tickers" % (sector, count))

expected_sectors = {
    "Automobiles": 3,
    "Banking & Financial Services": 5,
    "Energy & Power": 3,
    "FMCG / Consumer Staples": 3,
    "Information Technology": 4,
    "Infrastructure & Cement": 2,
    "Pharmaceuticals": 3,
}
sector_dict = dict(sectors)
check("Sector counts match expected",
      sector_dict == expected_sectors,
      "Got: %s" % sector_dict)

# =========================================================================
# Test 10: Data access layer functions
# =========================================================================
print("\n--- Test 10: Data Access Layer ---")

# get_ticker_info
all_tickers = get_ticker_info()
check("get_ticker_info() returns %d tickers" % len(all_tickers),
      len(all_tickers) == EXPECTED_TICKERS)

single = get_ticker_info("RELIANCE.NS")
check("get_ticker_info('RELIANCE.NS') returns 1 row",
      len(single) == 1 and single.iloc[0]["company_name"] == "Reliance Industries")

# get_calendar
cal = get_calendar("2020-01-01", "2020-12-31")
check("get_calendar(2020) returns trading days",
      len(cal) > 200, "Got %d days" % len(cal))

# get_close_matrix
matrix = get_close_matrix("2020-01-01", "2020-12-31")
check("get_close_matrix(2020) returns %dx%d matrix" % matrix.shape,
      matrix.shape[1] == EXPECTED_TICKERS,
      "Shape: %s" % str(matrix.shape))

conn.close()

# =========================================================================
# VERDICT
# =========================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("  Passed: %d" % passed)
print("  Failed: %d" % failed)

if failed == 0:
    print("\n  DATABASE VALIDATION CLEAR: All checks passed.")
    print("  DB matches the validated 4,109-day x 23-ticker dataset.")
else:
    print("\n  DATABASE VALIDATION FAILED: %d check(s) failed." % failed)

print("=" * 70)
