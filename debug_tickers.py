"""
Debug script: Investigate why TATAMOTORS.NS returns no data,
and inspect the 2 extra missing dates for MARUTI.NS.
"""
import yfinance as yf
import pandas as pd
import traceback

# =========================================================================
# PART 1: TATAMOTORS.NS DEBUG
# =========================================================================
print("=" * 70)
print("TATAMOTORS.NS DEBUG")
print("=" * 70)

# 1a. Ticker metadata
print("\n--- Ticker.info metadata ---")
try:
    t = yf.Ticker("TATAMOTORS.NS")
    info = t.info
    for key in ["symbol", "longName", "shortName", "exchange", "market",
                 "currency", "quoteType", "regularMarketPrice", "previousClose",
                 "marketState", "financialCurrency"]:
        print("  %s: %s" % (key, info.get(key, "N/A")))
except Exception as e:
    print("  Error getting info: %s" % e)
    traceback.print_exc()

# 1b. history() method - last 5 days
print("\n--- Ticker.history(period='5d') ---")
try:
    h = t.history(period="5d")
    print("  Rows returned: %d" % len(h))
    if len(h) > 0:
        print(h)
    else:
        print("  (empty DataFrame)")
except Exception as e:
    print("  Error: %s" % e)
    traceback.print_exc()

# 1c. yf.download - recent period, no auto_adjust
print("\n--- yf.download(period='1mo', auto_adjust=False) ---")
try:
    d = yf.download("TATAMOTORS.NS", period="1mo", auto_adjust=False, progress=False)
    print("  Rows: %d" % len(d))
    if len(d) > 0:
        print(d.tail())
    else:
        print("  (empty)")
except Exception as e:
    print("  Error: %s" % e)

# 1d. yf.download - specific date range
print("\n--- yf.download(start='2025-01-01', end='2025-06-01') ---")
try:
    d2 = yf.download("TATAMOTORS.NS", start="2025-01-01", end="2025-06-01",
                      auto_adjust=True, progress=False)
    print("  Rows: %d" % len(d2))
    if len(d2) > 0:
        print(d2.head(3))
        print("  ...")
        print(d2.tail(3))
    else:
        print("  (empty)")
except Exception as e:
    print("  Error: %s" % e)

# 1e. Try BSE variant
print("\n--- TATAMOTORS.BO (BSE fallback) ---")
try:
    d3 = yf.download("TATAMOTORS.BO", period="1mo", auto_adjust=True, progress=False)
    print("  Rows: %d" % len(d3))
    if len(d3) > 0:
        print(d3.tail())
    else:
        print("  (empty)")
except Exception as e:
    print("  Error: %s" % e)

# 1f. Try the ADR (TTM on NYSE)
print("\n--- TTM (Tata Motors ADR, NYSE) ---")
try:
    d4 = yf.download("TTM", period="1mo", auto_adjust=True, progress=False)
    print("  Rows: %d" % len(d4))
    if len(d4) > 0:
        print(d4.tail())
    else:
        print("  (empty)")
except Exception as e:
    print("  Error: %s" % e)

# 1g. Try fetching actions (splits/dividends) to see if ticker exists at all
print("\n--- Ticker.actions (splits/dividends history) ---")
try:
    actions = t.actions
    print("  Actions rows: %d" % len(actions))
    if len(actions) > 0:
        print(actions.tail(5))
    else:
        print("  (no actions)")
except Exception as e:
    print("  Error: %s" % e)

# 1h. Check the raw HTTP response
print("\n--- Raw download attempt with error capture ---")
try:
    import io, sys
    old_stderr = sys.stderr
    sys.stderr = buffer = io.StringIO()
    d5 = yf.download("TATAMOTORS.NS", start="2024-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    sys.stderr = old_stderr
    stderr_output = buffer.getvalue()
    print("  Rows: %d" % len(d5))
    if stderr_output:
        print("  stderr output:")
        for line in stderr_output.strip().split("\n"):
            print("    %s" % line)
except Exception as e:
    sys.stderr = old_stderr
    print("  Error: %s" % e)

print("\nyfinance version: %s" % yf.__version__)
print("pandas version: %s" % pd.__version__)


# =========================================================================
# PART 2: MARUTI.NS - Identify the exact 2 extra missing dates
# =========================================================================
print("\n\n" + "=" * 70)
print("MARUTI.NS - EXTRA GAP ANALYSIS")
print("=" * 70)

import pandas_market_calendars as mcal

raw = pd.read_pickle("data/raw/nifty_subset_raw.pkl")

# Get a reference ticker's missing dates (TCS has the standard 11)
ref = raw["TCS.NS"].dropna(how="all")
ref_dates = set(ref.index.date)
start, end = ref.index.min(), ref.index.max()

nse = mcal.get_calendar("NSE")
sched = nse.schedule(start_date=start, end_date=end)
expected = set(sched.index.date)
ref_missing = expected - ref_dates

# Get MARUTI's missing dates
maruti = raw["MARUTI.NS"].dropna(how="all")
maruti_dates = set(maruti.index.date)
maruti_start, maruti_end = maruti.index.min(), maruti.index.max()
maruti_expected = set(nse.schedule(start_date=maruti_start, end_date=maruti_end).index.date)
maruti_missing = maruti_expected - maruti_dates

# The extra gaps unique to MARUTI
extra_gaps = sorted(maruti_missing - ref_missing)
print("\nMARUTI.NS has %d missing days vs TCS.NS's %d" % (len(maruti_missing), len(ref_missing)))
print("Extra gaps unique to MARUTI.NS:")
for d in extra_gaps:
    print("  %s (%s)" % (d, pd.Timestamp(d).day_name()))
    # Check if TCS has data for this date
    if d in ref_dates:
        tcs_close = ref.loc[ref.index.date == d, "Close"]
        print("    TCS.NS had data on this date (Close: %s) -- MARUTI is missing" % tcs_close.values)
    else:
        print("    TCS.NS also missing -- this is a shared gap")

# Also check: dates MARUTI has but TCS doesn't
maruti_extra_vs_tcs = sorted(maruti_dates - ref_dates)
if maruti_extra_vs_tcs:
    print("\nDates MARUTI.NS has but TCS.NS doesn't:")
    for d in maruti_extra_vs_tcs:
        print("  %s (%s)" % (d, pd.Timestamp(d).day_name()))

# Check MARUTI's extra dates vs calendar
maruti_extra_vs_cal = sorted(maruti_dates - maruti_expected)
ref_extra_vs_cal = sorted(ref_dates - expected)
maruti_only_extra = sorted(set(ref_extra_vs_cal) - set(maruti_extra_vs_cal))
print("\nExtra dates TCS has that MARUTI doesn't:")
for d in maruti_only_extra:
    print("  %s (%s)" % (d, pd.Timestamp(d).day_name()))

# Direct date range comparison
print("\nMARUTI date range: %s to %s" % (maruti_start.date(), maruti_end.date()))
print("TCS    date range: %s to %s" % (ref.index.min().date(), ref.index.max().date()))
print("MARUTI rows: %d, TCS rows: %d, diff: %d" % (len(maruti_dates), len(ref_dates), len(maruti_dates) - len(ref_dates)))
