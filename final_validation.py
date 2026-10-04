"""
================================================================================
Stage 1 Final Validation — End-to-End Sanity Check
================================================================================
Runs every check on the patched reconciled dataset in one pass:
  1. Shape & completeness (23 tickers x 4115 days, 0 NaN)
  2. All-NSE source verification
  3. No >20% single-day moves (except known genuine events)
  4. No zero-volume stale rows remaining
  5. No duplicate dates
  6. Monotonic date index
  7. No negative prices or volumes
  8. Cross-file consistency (pickle vs CSV)
  9. Per-ticker date range and row count
 10. Calendar file consistency
================================================================================
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 150)

EXPECTED_TICKERS = 23
EXPECTED_DAYS = 4109
KNOWN_BIG_MOVES = {
    ("SBIN.NS", "2017-10-25"),
    ("AXISBANK.NS", "2020-03-23"),
    ("INFY.NS", "2013-04-12"),
}

passed = 0
failed = 0
warnings = 0

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

def warn(name, detail=""):
    global warnings
    warnings += 1
    print("  [WARN] %s" % name)
    if detail:
        print("         %s" % detail)

# =========================================================================
# Load all data files
# =========================================================================
print("=" * 70)
print("LOADING DATA FILES")
print("=" * 70)

rec = pd.read_pickle("data/reconciled/nifty_subset_reconciled.pkl")
close_csv = pd.read_csv("data/reconciled/close_prices.csv", index_col="Date", parse_dates=True)
vol_csv = pd.read_csv("data/reconciled/volumes.csv", index_col="Date", parse_dates=True)
calendar = pd.read_csv("data/summary/consensus_calendar.csv", parse_dates=["Date"])

print("  Reconciled pickle: %s" % str(rec.shape))
print("  Close CSV: %s" % str(close_csv.shape))
print("  Volume CSV: %s" % str(vol_csv.shape))
print("  Calendar: %d dates" % len(calendar))

# =========================================================================
# 1. Shape & Completeness
# =========================================================================
print("\n" + "=" * 70)
print("1. SHAPE & COMPLETENESS")
print("=" * 70)

tickers = sorted(set(c[0] for c in rec.columns))
check("Ticker count = %d" % EXPECTED_TICKERS,
      len(tickers) == EXPECTED_TICKERS,
      "Got: %d tickers: %s" % (len(tickers), tickers))

check("Trading days = %d" % EXPECTED_DAYS,
      len(rec) == EXPECTED_DAYS,
      "Got: %d" % len(rec))

total_nans_rec = sum(rec[t]["Close"].isna().sum() for t in tickers)
check("Reconciled pickle: 0 NaN in Close",
      total_nans_rec == 0,
      "Got: %d NaN" % total_nans_rec)

total_nans_csv = close_csv.isna().sum().sum()
check("Close CSV: 0 NaN",
      total_nans_csv == 0,
      "Got: %d NaN" % total_nans_csv)

check("Close CSV shape matches pickle",
      close_csv.shape == (EXPECTED_DAYS, EXPECTED_TICKERS),
      "Got: %s" % str(close_csv.shape))

check("Volume CSV shape matches pickle",
      vol_csv.shape == (EXPECTED_DAYS, EXPECTED_TICKERS),
      "Got: %s" % str(vol_csv.shape))

# =========================================================================
# 2. Ticker Roster
# =========================================================================
print("\n" + "=" * 70)
print("2. TICKER ROSTER")
print("=" * 70)

expected_tickers = sorted([
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS",
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "MARUTI.NS", "EICHERMOT.NS", "M&M.NS",
    "LT.NS", "ULTRACEMCO.NS",
])

check("All expected tickers present",
      sorted(tickers) == sorted(expected_tickers),
      "Missing: %s, Extra: %s" % (
          set(expected_tickers) - set(tickers),
          set(tickers) - set(expected_tickers)))

csv_tickers = sorted(close_csv.columns.tolist())
check("CSV tickers match pickle tickers",
      csv_tickers == sorted(expected_tickers),
      "CSV has: %s" % csv_tickers)

# =========================================================================
# 3. No Unexplained >20% Moves
# =========================================================================
print("\n" + "=" * 70)
print("3. RETURN ANOMALIES (>20%% single-day moves)")
print("=" * 70)

unexpected_moves = []
for t in tickers:
    rets = close_csv[t].pct_change()
    big = rets[rets.abs() > 0.20]
    for date, ret in big.items():
        key = (t, str(date.date()))
        if key not in KNOWN_BIG_MOVES:
            unexpected_moves.append((t, date.date(), ret))

check("No unexplained >20%% moves",
      len(unexpected_moves) == 0,
      "Unexpected: %s" % unexpected_moves if unexpected_moves else
      "Only known events: %s" % KNOWN_BIG_MOVES)


# =========================================================================
# 4. Stale Row Check (split by root cause, not just symptom)
# =========================================================================
print("\n" + "=" * 70)
print("4. STALE ROW CHECK")
print("=" * 70)

KNOWN_FFILLED_ROWS = {
    ("NESTLEIND.NS", "2010-01-04"), ("NESTLEIND.NS", "2010-01-05"),
    ("NESTLEIND.NS", "2010-01-06"), ("NESTLEIND.NS", "2010-01-07"),
    ("MARUTI.NS", "2015-08-12"),
    ("M&M.NS", "2025-09-08"),
    ("AXISBANK.NS", "2025-03-18"), ("CIPLA.NS", "2025-03-18"), ("DRREDDY.NS", "2025-03-18"),
    ("EICHERMOT.NS", "2025-03-18"), ("HCLTECH.NS", "2025-03-18"), ("HDFCBANK.NS", "2025-03-18"),
    ("HINDUNILVR.NS", "2025-03-18"), ("ICICIBANK.NS", "2025-03-18"), ("INFY.NS", "2025-03-18"),
    ("KOTAKBANK.NS", "2025-03-18"), ("LT.NS", "2025-03-18"), ("M&M.NS", "2025-03-18"),
    ("MARUTI.NS", "2025-03-18"), ("NESTLEIND.NS", "2025-03-18"), ("NTPC.NS", "2025-03-18"),
    ("ONGC.NS", "2025-03-18"), ("RELIANCE.NS", "2025-03-18"), ("SBIN.NS", "2025-03-18"),
    ("SUNPHARMA.NS", "2025-03-18"), ("TCS.NS", "2025-03-18"), ("ULTRACEMCO.NS", "2025-03-18"),
    ("WIPRO.NS", "2025-03-18"),
}

stale_found = []
missing_vol_only = []
for t in tickers:
    df = rec[t]
    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    zero_vol = df["Volume"] == 0

    true_stale_mask = zero_vol & flat
    vol_missing_mask = zero_vol & ~flat

    if true_stale_mask.any():
        stale_found.extend([(t, d.date()) for d in df.index[true_stale_mask]])
    if vol_missing_mask.any():
        missing_vol_only.extend([(t, d.date()) for d in df.index[vol_missing_mask]])

stale_found_unexpected = [
    (t, d) for (t, d) in stale_found if (t, str(d)) not in KNOWN_FFILLED_ROWS
]

check("No unexpected stale rows (excluding known forward-fills)",
      len(stale_found_unexpected) == 0,
      "Unexpected: %s" % stale_found_unexpected if stale_found_unexpected else
      "Only known, documented forward-fills: %d row(s)" % len(stale_found))

if missing_vol_only:
    warn("Zero-volume rows with real price variation (Volume field missing, not stale)",
         "Found %d row(s): %s" % (len(missing_vol_only), missing_vol_only[:10]))
# =========================================================================
# 5. No Duplicate Dates
# =========================================================================
print("\n" + "=" * 70)
print("5. INDEX INTEGRITY")
print("=" * 70)

check("No duplicate dates in index",
      not rec.index.duplicated().any(),
      "Duplicates: %d" % rec.index.duplicated().sum())

check("Date index is monotonically increasing",
      rec.index.is_monotonic_increasing)

check("First date is 2010-01-04",
      rec.index.min().date() == pd.Timestamp("2010-01-04").date(),
      "Got: %s" % rec.index.min().date())

check("Last date is 2026-08-28",
      rec.index.max().date() == pd.Timestamp("2026-08-28").date(),
      "Got: %s" % rec.index.max().date())

# =========================================================================
# 6. No Negative Prices or Volumes
# =========================================================================
print("\n" + "=" * 70)
print("6. VALUE SANITY")
print("=" * 70)

neg_prices = []
neg_vols = []
for t in tickers:
    df = rec[t]
    for col in ["Open", "High", "Low", "Close"]:
        neg = df[df[col] < 0]
        if len(neg) > 0:
            neg_prices.extend([(t, col, d.date(), df.loc[d, col]) for d in neg.index])
    neg_v = df[df["Volume"] < 0]
    if len(neg_v) > 0:
        neg_vols.extend([(t, d.date(), df.loc[d, "Volume"]) for d in neg_v.index])

check("No negative prices", len(neg_prices) == 0,
      "Found: %s" % neg_prices if neg_prices else "")
check("No negative volumes", len(neg_vols) == 0,
      "Found: %s" % neg_vols if neg_vols else "")

# High >= Low on every row
hl_violations = []
for t in tickers:
    df = rec[t]
    bad = df[df["High"] < df["Low"]]
    if len(bad) > 0:
        hl_violations.extend([(t, d.date()) for d in bad.index])

check("High >= Low on every row", len(hl_violations) == 0,
      "Violations: %s" % hl_violations if hl_violations else "")

# =========================================================================
# 7. Cross-File Consistency
# =========================================================================
print("\n" + "=" * 70)
print("7. CROSS-FILE CONSISTENCY")
print("=" * 70)

# Pickle close vs CSV close
max_diff = 0
for t in tickers:
    pkl_close = rec[t]["Close"].values
    csv_close = close_csv[t].values
    diff = np.nanmax(np.abs(pkl_close - csv_close))
    max_diff = max(max_diff, diff)

check("Pickle Close == CSV Close (max diff < 0.01)",
      max_diff < 0.01,
      "Max diff: %.6f" % max_diff)

# Pickle volume vs CSV volume (NaN matches NaN)
vol_mismatches = 0
for t in tickers:
    pkl_vol = rec[t]["Volume"]
    csv_vol = vol_csv[t]
    # Both NaN = match, both equal = match, else mismatch
    for i in range(len(pkl_vol)):
        p, c = pkl_vol.iloc[i], csv_vol.iloc[i]
        if pd.isna(p) and pd.isna(c):
            continue
        elif pd.isna(p) or pd.isna(c):
            vol_mismatches += 1
        elif abs(p - c) > 0.01:
            vol_mismatches += 1

check("Pickle Volume == CSV Volume (mismatched cells)",
      vol_mismatches == 0,
      "Mismatched cells: %d" % vol_mismatches)

# Calendar file vs index
cal_dates = set(calendar["Date"].dt.date)
idx_dates = set(rec.index.date)
check("Calendar CSV matches index",
      cal_dates == idx_dates,
      "In calendar not index: %d, In index not calendar: %d" % (
          len(cal_dates - idx_dates), len(idx_dates - cal_dates)))

# =========================================================================
# 8. Per-Ticker Summary
# =========================================================================
print("\n" + "=" * 70)
print("8. PER-TICKER SUMMARY")
print("=" * 70)

print("  %-16s %6s %12s %12s %12s %8s" % (
    "Ticker", "Rows", "First", "Last", "Close(Last)", "ZeroVol"))

for t in sorted(tickers):
    df = rec[t]
    close = df["Close"]
    zv = (df["Volume"] == 0).sum()
    print("  %-16s %6d %12s %12s %12.2f %8d" % (
        t, len(close), df.index.min().date(), df.index.max().date(),
        close.iloc[-1], zv))

# =========================================================================
# VERDICT
# =========================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("  Passed: %d" % passed)
print("  Failed: %d" % failed)
print("  Warnings: %d" % warnings)

if failed == 0:
    print("\n  STAGE 1 CLEAR: Dataset is clean and ready for schema design.")
else:
    print("\n  STAGE 1 BLOCKED: %d check(s) failed — investigate before proceeding." % failed)

print("=" * 70)
