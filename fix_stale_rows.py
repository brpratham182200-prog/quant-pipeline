"""
================================================================================
Dataset-Wide Stale Row Fix
================================================================================
The validation found 165 stale rows (zero volume + flat OHLC) across the dataset.

These fall into 3 categories:

A) 6 dates where ALL 23 tickers are stale → DROP from calendar entirely
   (These are NSE holidays Yahoo emits with stale fills)
   - 2014-04-24, 2014-10-15, 2026-01-15, 2026-05-01, 2026-05-28, 2026-06-26

B) 1 date where 22/23 tickers are stale, 1 has real data → KEEP, forward-fill
   - 2025-03-18: ITC.NS has real data (13.8M volume), 22 others stale

C) Single-ticker stale dates → forward-fill that ticker
   - NESTLEIND.NS: 2010-01-04 to 2010-01-07 (4 rows, pre-trading placeholder)
   - MARUTI.NS: 2015-08-12 (1 row)
   - M&M.NS: 2025-09-08 (1 row)

Note: 2014-04-24 shows ONGC as "real" in the 22-ticker count, but inspection
shows it's also stale (V=0, flat OHLC) — floating-point comparison issue.
So it's effectively 23/23 → drop.
================================================================================
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 150)

# =========================================================================
# 1. Load and classify
# =========================================================================
print("=" * 70)
print("STEP 1: Load and classify stale rows")
print("=" * 70)

rec = pd.read_pickle("data/reconciled/nifty_subset_reconciled.pkl")
tickers = sorted(set(c[0] for c in rec.columns))
print("  Starting shape: %d dates x %d tickers" % (len(rec), len(tickers)))

# Find all stale rows per date
from collections import Counter
stale_by_date = Counter()
stale_detail = {}

for t in tickers:
    df = rec[t]
    # Use tolerance for float comparison
    mask = (df["Volume"] == 0) & \
           (abs(df["Open"] - df["High"]) < 0.01) & \
           (abs(df["High"] - df["Low"]) < 0.01) & \
           (abs(df["Low"] - df["Close"]) < 0.01)
    for d in df.index[mask]:
        stale_by_date[d.date()] += 1
        if d.date() not in stale_detail:
            stale_detail[d.date()] = []
        stale_detail[d.date()].append(t)

print("  Total stale dates: %d" % len(stale_by_date))

# Category A: drop dates (all or nearly all tickers stale)
DROP_DATES = []
FFILL_DATES = {}  # date -> list of tickers to forward-fill

for d, count in sorted(stale_by_date.items()):
    # Check if any ticker has REAL data on this date
    real_tickers = []
    for t in tickers:
        if t not in stale_detail.get(d, []):
            row = rec.loc[pd.Timestamp(d), (t, "Volume")]
            if row > 0:
                real_tickers.append(t)

    if count >= 22 and len(real_tickers) == 0:
        # All tickers stale → drop the date
        DROP_DATES.append(d)
        print("  %s: %d/%d stale, 0 real -> DROP" % (d, count, len(tickers)))
    elif count >= 22 and len(real_tickers) > 0:
        # Most stale but some real → keep, forward-fill the stale ones
        stale_tickers = stale_detail[d]
        FFILL_DATES[d] = stale_tickers
        print("  %s: %d/%d stale, %d real (%s) -> KEEP, ffill %d" % (
            d, count, len(tickers), len(real_tickers),
            ", ".join(real_tickers), len(stale_tickers)))
    else:
        # Single-ticker stale → forward-fill that ticker
        FFILL_DATES[d] = stale_detail[d]
        print("  %s: %d/%d stale -> FFILL %s" % (
            d, count, len(tickers), stale_detail[d]))

# =========================================================================
# 2. Drop full-stale dates from calendar
# =========================================================================
print("\n" + "=" * 70)
print("STEP 2: Dropping %d full-stale dates" % len(DROP_DATES))
print("=" * 70)

drop_timestamps = [pd.Timestamp(d) for d in DROP_DATES]
rec = rec.drop(index=drop_timestamps, errors="ignore")
print("  New shape: %d dates" % len(rec))
for d in DROP_DATES:
    print("  Dropped: %s (%s)" % (d, pd.Timestamp(d).day_name()))

# =========================================================================
# 3. Forward-fill remaining stale rows
# =========================================================================
print("\n" + "=" * 70)
print("STEP 3: Forward-filling remaining stale rows")
print("=" * 70)

ffill_count = 0
for d, stale_tickers in sorted(FFILL_DATES.items()):
    ts = pd.Timestamp(d)
    if ts not in rec.index:
        continue  # was already dropped
    for t in stale_tickers:
        idx = rec.index.get_loc(ts)
        if idx == 0:
            # Backfill from first real row for this ticker
            df = rec[t]
            non_stale = df[(df["Volume"] > 0) | (abs(df["Open"] - df["Close"]) > 0.01)]
            if len(non_stale) > 0:
                fill_price = non_stale.iloc[0]["Close"]
            else:
                continue
        else:
            fill_price = rec.iloc[idx - 1][(t, "Close")]

        for col in ["Open", "High", "Low", "Close"]:
            rec.loc[ts, (t, col)] = fill_price
        # Volume stays 0
        ffill_count += 1

print("  Forward-filled %d ticker-date pairs" % ffill_count)

# =========================================================================
# 4. Save everything
# =========================================================================
print("\n" + "=" * 70)
print("STEP 4: Saving all outputs")
print("=" * 70)

# Save reconciled pickle
rec.to_pickle("data/reconciled/nifty_subset_reconciled.pkl")
print("  [SAVED] nifty_subset_reconciled.pkl")

# Rebuild close_prices.csv
close_matrix = pd.DataFrame(index=rec.index)
close_matrix.index.name = "Date"
vol_matrix = pd.DataFrame(index=rec.index)
vol_matrix.index.name = "Date"

for t in tickers:
    close_matrix[t] = rec[t]["Close"]
    vol_matrix[t] = rec[t]["Volume"]

close_matrix.to_csv("data/reconciled/close_prices.csv")
vol_matrix.to_csv("data/reconciled/volumes.csv")
print("  [SAVED] close_prices.csv, volumes.csv")

# Update consensus calendar
cal = pd.DataFrame({"Date": rec.index.date})
cal.to_csv("data/summary/consensus_calendar.csv", index=False)
print("  [SAVED] consensus_calendar.csv")

# =========================================================================
# 5. Re-run full validation
# =========================================================================
print("\n" + "=" * 70)
print("STEP 5: Full validation on fixed dataset")
print("=" * 70)

rec = pd.read_pickle("data/reconciled/nifty_subset_reconciled.pkl")
close_csv = pd.read_csv("data/reconciled/close_prices.csv", index_col="Date", parse_dates=True)

tickers = sorted(set(c[0] for c in rec.columns))
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print("  [PASS] %s" % name)
    else:
        failed += 1
        print("  [FAIL] %s%s" % (name, " — " + detail if detail else ""))

# Shape
check("Ticker count = 23", len(tickers) == 23)
check("Trading days", len(rec) > 0, "Got %d" % len(rec))

# NaN
nan_pkl = sum(rec[t]["Close"].isna().sum() for t in tickers)
nan_csv = close_csv.isna().sum().sum()
check("0 NaN in pickle Close", nan_pkl == 0, "Got %d" % nan_pkl)
check("0 NaN in CSV", nan_csv == 0, "Got %d" % nan_csv)

# Stale rows
stale_remaining = 0
for t in tickers:
    df = rec[t]
    mask = (df["Volume"] == 0) & \
           (abs(df["Open"] - df["High"]) < 0.01) & \
           (abs(df["High"] - df["Low"]) < 0.01) & \
           (abs(df["Low"] - df["Close"]) < 0.01)
    stale_remaining += mask.sum()
check("0 stale rows (zero-vol + flat OHLC)", stale_remaining == 0,
      "Got %d" % stale_remaining)

# >20% moves
KNOWN = {("SBIN.NS", "2017-10-25"), ("AXISBANK.NS", "2020-03-23"), ("INFY.NS", "2013-04-12")}
unexpected = []
for t in tickers:
    rets = close_csv[t].pct_change()
    big = rets[rets.abs() > 0.20]
    for date, ret in big.items():
        if (t, str(date.date())) not in KNOWN:
            unexpected.append((t, date.date(), "%.1f%%" % (ret * 100)))
check("No unexplained >20%% moves", len(unexpected) == 0,
      "Got %s" % unexpected)

# Index integrity
check("No duplicate dates", not rec.index.duplicated().any())
check("Monotonically increasing", rec.index.is_monotonic_increasing)
check("First date = 2010-01-04", rec.index.min().date() == pd.Timestamp("2010-01-04").date())

# No negative prices
has_neg = False
for t in tickers:
    for col in ["Open", "High", "Low", "Close"]:
        if (rec[t][col] < 0).any():
            has_neg = True
check("No negative prices", not has_neg)

# High >= Low
hl_ok = True
for t in tickers:
    if (rec[t]["High"] < rec[t]["Low"]).any():
        hl_ok = False
check("High >= Low everywhere", hl_ok)

# Cross-file consistency
max_diff = 0
for t in tickers:
    diff = np.nanmax(np.abs(rec[t]["Close"].values - close_csv[t].values))
    max_diff = max(max_diff, diff)
check("Pickle == CSV (max diff < 0.01)", max_diff < 0.01)

# Calendar
cal = pd.read_csv("data/summary/consensus_calendar.csv", parse_dates=["Date"])
check("Calendar matches index", set(cal["Date"].dt.date) == set(rec.index.date))

# Zero-vol check: if vol=0, close should equal prior close
zero_vol_mismatch = 0
for t in tickers:
    df = rec[t]
    for i in range(1, len(df)):
        if df.iloc[i]["Volume"] == 0:
            if abs(df.iloc[i]["Close"] - df.iloc[i-1]["Close"]) > 0.01:
                zero_vol_mismatch += 1
check("Zero-volume rows: close == prior close", zero_vol_mismatch == 0,
      "Got %d mismatches" % zero_vol_mismatch)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("PER-TICKER SUMMARY")
print("=" * 70)
print("  %-16s %6s %12s %12s %12s %8s" % (
    "Ticker", "Rows", "First", "Last", "Close(Last)", "ZeroVol"))
for t in sorted(tickers):
    df = rec[t]
    zv = (df["Volume"] == 0).sum()
    print("  %-16s %6d %12s %12s %12.2f %8d" % (
        t, len(df), df.index.min().date(), df.index.max().date(),
        df["Close"].iloc[-1], zv))

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("  Passed: %d" % passed)
print("  Failed: %d" % failed)
if failed == 0:
    print("\n  STAGE 1 CLEAR: Dataset ready for schema design.")
else:
    print("\n  STAGE 1 BLOCKED: %d check(s) failed." % failed)
print("=" * 70)
