"""
================================================================================
Calendar Reconciliation & Ticker Fix Script
================================================================================
1. Downloads EICHER.NS (Eicher Motors) as TATAMOTORS replacement
2. Patches the raw pickle and per-ticker CSVs
3. Builds a consensus intersection calendar from all 22 clean tickers
4. Aligns all tickers to the consensus calendar
5. Saves the reconciled dataset
================================================================================
"""
import yfinance as yf
import pandas as pd
import os

# =============================================================================
# CONFIG
# =============================================================================
RAW_PICKLE = "data/raw/nifty_subset_raw.pkl"
PER_TICKER_DIR = "data/per_ticker"
SUMMARY_DIR = "data/summary"
RECONCILED_DIR = "data/reconciled"
os.makedirs(RECONCILED_DIR, exist_ok=True)

TICKER_CATEGORIES = {
    "Banking & Financial Services": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS"],
    "Information Technology": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS"],
    "Energy & Power": ["RELIANCE.NS","ONGC.NS","NTPC.NS"],
    "FMCG / Consumer Staples": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
    "Automobiles": ["MARUTI.NS","EICHERMOT.NS","M&M.NS"],  # EICHER replaces TATAMOTORS
    "Infrastructure & Cement": ["LT.NS","ULTRACEMCO.NS"],
}
ALL_TICKERS = [t for tickers in TICKER_CATEGORIES.values() for t in tickers]

# =============================================================================
# STEP 1: Download EICHER.NS
# =============================================================================
print("=" * 70)
print("STEP 1: Downloading EICHER.NS (Eicher Motors)")
print("=" * 70)

eicher = yf.download(
    "EICHERMOT.NS",
    start="2010-01-01",
    end="2026-08-31",
    interval="1d",
    auto_adjust=True,
    progress=True,
)

print("  Rows downloaded: %d" % len(eicher))
print("  Date range: %s to %s" % (eicher.index.min().date(), eicher.index.max().date()))
print("  Head:")
print(eicher.head(3).to_string())
print("  Tail:")
print(eicher.tail(3).to_string())

# Save per-ticker CSV
csv_path = os.path.join(PER_TICKER_DIR, "EICHER_NS.csv")
eicher.to_csv(csv_path)
print("\n  Saved -> %s" % csv_path)

# =============================================================================
# STEP 2: Update raw pickle — remove TATAMOTORS, add EICHER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Updating raw pickle")
print("=" * 70)

raw = pd.read_pickle(RAW_PICKLE)

# Remove TATAMOTORS columns
tata_cols = [c for c in raw.columns if "TATAMOTORS" in str(c)]
if tata_cols:
    raw = raw.drop(columns=tata_cols)
    print("  Removed %d TATAMOTORS columns" % len(tata_cols))

# Add EICHER columns
for col in eicher.columns:
    if isinstance(col, tuple):
        field = col[0]
    else:
        field = col
    raw[("EICHERMOT.NS", field)] = eicher[col] if not isinstance(col, tuple) else eicher[col]

raw.to_pickle(RAW_PICKLE)
print("  Added EICHERMOT.NS columns to pickle")
print("  New pickle shape: %s" % str(raw.shape))

# Remove TATAMOTORS CSV
tata_csv = os.path.join(PER_TICKER_DIR, "TATAMOTORS_NS.csv")
if os.path.exists(tata_csv):
    os.remove(tata_csv)
    print("  Removed %s" % tata_csv)

# =============================================================================
# STEP 3: Build consensus intersection calendar
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Building consensus intersection calendar")
print("=" * 70)

# Reload to get fresh state
raw = pd.read_pickle(RAW_PICKLE)

# Collect each ticker's trading dates
ticker_date_sets = {}
for ticker in ALL_TICKERS:
    try:
        df = raw[ticker].dropna(how="all")
        if len(df) > 0:
            ticker_date_sets[ticker] = set(df.index.date)
            print("  %s: %d dates" % (ticker, len(ticker_date_sets[ticker])))
        else:
            print("  %s: NO DATA (skipped)" % ticker)
    except KeyError:
        print("  %s: NOT IN PICKLE (skipped)" % ticker)

# Intersection: dates where ALL tickers have data
all_sets = list(ticker_date_sets.values())
consensus_dates = sorted(set.intersection(*all_sets))

print("\n  Tickers included: %d" % len(ticker_date_sets))
print("  Consensus calendar size: %d trading days" % len(consensus_dates))
print("  Date range: %s to %s" % (consensus_dates[0], consensus_dates[-1]))

# Show what gets dropped for each ticker
print("\n  Dates dropped per ticker (not in consensus):")
for ticker, dates in ticker_date_sets.items():
    dropped = sorted(dates - set(consensus_dates))
    if dropped:
        print("    %s: %d dates dropped" % (ticker, len(dropped)))
        for d in dropped[:5]:
            print("      %s (%s)" % (d, pd.Timestamp(d).day_name()))
        if len(dropped) > 5:
            print("      ... and %d more" % (len(dropped) - 5))

# Save consensus calendar
consensus_df = pd.DataFrame({"Date": consensus_dates})
cal_path = os.path.join(SUMMARY_DIR, "consensus_calendar.csv")
consensus_df.to_csv(cal_path, index=False)
print("\n  Consensus calendar saved -> %s" % cal_path)

# =============================================================================
# STEP 4: Build reconciled dataset aligned to consensus calendar
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Building reconciled dataset")
print("=" * 70)

consensus_index = pd.DatetimeIndex(consensus_dates)

# Build a clean close-price matrix (tickers x dates)
close_matrix = pd.DataFrame(index=consensus_index)
close_matrix.index.name = "Date"

volume_matrix = pd.DataFrame(index=consensus_index)
volume_matrix.index.name = "Date"

for ticker in ALL_TICKERS:
    if ticker not in ticker_date_sets:
        continue
    try:
        df = raw[ticker].dropna(how="all")
        # Align to consensus calendar
        aligned = df.reindex(consensus_index)

        close_col = aligned["Close"] if "Close" in aligned.columns else None
        vol_col = aligned["Volume"] if "Volume" in aligned.columns else None

        if close_col is not None:
            close_matrix[ticker] = close_col
        if vol_col is not None:
            volume_matrix[ticker] = vol_col

        nulls = close_matrix[ticker].isna().sum() if ticker in close_matrix.columns else "N/A"
        print("  %s: aligned, %s NaN close values" % (ticker, nulls))
    except Exception as e:
        print("  %s: Error - %s" % (ticker, e))

# Save reconciled data
close_path = os.path.join(RECONCILED_DIR, "close_prices.csv")
close_matrix.to_csv(close_path)
print("\n  Close price matrix saved -> %s (%d x %d)" % (
    close_path, close_matrix.shape[0], close_matrix.shape[1]))

vol_path = os.path.join(RECONCILED_DIR, "volumes.csv")
volume_matrix.to_csv(vol_path)
print("  Volume matrix saved -> %s (%d x %d)" % (
    vol_path, volume_matrix.shape[0], volume_matrix.shape[1]))

# Also save a full reconciled pickle with all OHLCV
reconciled = {}
for ticker in ALL_TICKERS:
    if ticker not in ticker_date_sets:
        continue
    try:
        df = raw[ticker].dropna(how="all")
        reconciled[ticker] = df.reindex(consensus_index)
    except Exception:
        pass

reconciled_raw = pd.concat(reconciled, axis=1)
reconciled_path = os.path.join(RECONCILED_DIR, "nifty_subset_reconciled.pkl")
reconciled_raw.to_pickle(reconciled_path)
print("  Reconciled pickle saved -> %s" % reconciled_path)

# =============================================================================
# STEP 5: Final validation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Final validation")
print("=" * 70)

# Check for any NaN in the close matrix
nan_counts = close_matrix.isna().sum()
total_nans = nan_counts.sum()
print("\n  Total NaN values in close price matrix: %d" % total_nans)
if total_nans > 0:
    print("  Per-ticker NaN counts:")
    for ticker, count in nan_counts[nan_counts > 0].items():
        print("    %s: %d NaN values" % (ticker, count))
else:
    print("  All tickers have complete data on all consensus dates!")

print("\n  Final dataset summary:")
print("    Tickers: %d" % close_matrix.shape[1])
print("    Trading days: %d" % close_matrix.shape[0])
print("    Date range: %s to %s" % (close_matrix.index.min().date(), close_matrix.index.max().date()))
print("    Data completeness: %.1f%%" % (
    (1 - total_nans / (close_matrix.shape[0] * close_matrix.shape[1])) * 100))

# List all tickers in the final dataset
print("\n  Tickers in final dataset:")
for sector, tickers in TICKER_CATEGORIES.items():
    print("    %s:" % sector)
    for t in tickers:
        if t in close_matrix.columns:
            latest = close_matrix[t].dropna().iloc[-1] if close_matrix[t].dropna().shape[0] > 0 else "N/A"
            print("      %s  (latest close: %.2f)" % (t, latest))
        else:
            print("      %s  (NOT IN DATASET)" % t)

print("\n" + "=" * 70)
print("RECONCILIATION COMPLETE")
print("=" * 70)
print("  Reconciled pickle : %s" % reconciled_path)
print("  Close prices CSV  : %s" % close_path)
print("  Volumes CSV       : %s" % vol_path)
print("  Consensus calendar: %s" % cal_path)
print("=" * 70)
