"""
================================================================================
Load Reconciled Dataset into SQLite Database
================================================================================
Reads the validated reconciled pickle (23 tickers x 4,109 days) and loads it
into the quant.db SQLite database with full metadata, data quality annotations,
trading calendar, and errata logging.

Source data: data/reconciled/nifty_subset_reconciled.pkl
Target DB:   db/quant.db
================================================================================
"""
import sqlite3
import pandas as pd
import os

# =============================================================================
# CONFIG
# =============================================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "quant.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "schema.sql")
PICKLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reconciled", "nifty_subset_reconciled.pkl")
CALENDAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "summary", "consensus_calendar.csv")

# =============================================================================
# FULL TICKER METADATA — all 23 tickers
# Sourced from TICKER_CATEGORIES in download_nifty_data.py / reconcile_calendar.py
# =============================================================================
TICKER_META = {
    # Banking & Financial Services (5)
    "HDFCBANK.NS":   ("HDFC Bank",                    "Banking & Financial Services", "NSE"),
    "ICICIBANK.NS":  ("ICICI Bank",                   "Banking & Financial Services", "NSE"),
    "SBIN.NS":       ("State Bank of India",          "Banking & Financial Services", "NSE"),
    "KOTAKBANK.NS":  ("Kotak Mahindra Bank",          "Banking & Financial Services", "NSE"),
    "AXISBANK.NS":   ("Axis Bank",                    "Banking & Financial Services", "NSE"),

    # Information Technology (4)
    "TCS.NS":        ("Tata Consultancy Services",    "Information Technology",       "NSE"),
    "INFY.NS":       ("Infosys",                      "Information Technology",       "NSE"),
    "WIPRO.NS":      ("Wipro",                        "Information Technology",       "NSE"),
    "HCLTECH.NS":    ("HCL Technologies",             "Information Technology",       "NSE"),

    # Energy & Power (3)
    "RELIANCE.NS":   ("Reliance Industries",          "Energy & Power",               "NSE"),
    "ONGC.NS":       ("Oil & Natural Gas Corporation","Energy & Power",               "NSE"),
    "NTPC.NS":       ("NTPC Limited",                 "Energy & Power",               "NSE"),

    # FMCG / Consumer Staples (3)
    "HINDUNILVR.NS": ("Hindustan Unilever",           "FMCG / Consumer Staples",      "NSE"),
    "ITC.NS":        ("ITC Limited",                  "FMCG / Consumer Staples",      "NSE"),
    "NESTLEIND.NS":  ("Nestlé India",                 "FMCG / Consumer Staples",      "NSE"),

    # Pharmaceuticals (3)
    "SUNPHARMA.NS":  ("Sun Pharmaceutical",           "Pharmaceuticals",              "NSE"),
    "DRREDDY.NS":    ("Dr. Reddy's Laboratories",     "Pharmaceuticals",              "NSE"),
    "CIPLA.NS":      ("Cipla",                        "Pharmaceuticals",              "NSE"),

    # Automobiles (3)
    "MARUTI.NS":     ("Maruti Suzuki",                "Automobiles",                  "NSE"),
    "EICHERMOT.NS":  ("Eicher Motors",                "Automobiles",                  "NSE"),
    "M&M.NS":        ("Mahindra & Mahindra",          "Automobiles",                  "NSE"),

    # Infrastructure & Cement (2)
    "LT.NS":         ("Larsen & Toubro",              "Infrastructure & Cement",      "NSE"),
    "ULTRACEMCO.NS": ("UltraTech Cement",             "Infrastructure & Cement",      "NSE"),
}

# =============================================================================
# KNOWN DATA-QUALITY ISSUES — from final_validation.py errata
# Complete set of all 28 known forward-filled rows
# =============================================================================
KNOWN_FFILLED_ROWS = {
    # Category C: Single-ticker stale dates
    ("NESTLEIND.NS", "2010-01-04"), ("NESTLEIND.NS", "2010-01-05"),
    ("NESTLEIND.NS", "2010-01-06"), ("NESTLEIND.NS", "2010-01-07"),
    ("MARUTI.NS",    "2015-08-12"),
    ("M&M.NS",       "2025-09-08"),
    # Category B: 2025-03-18 — 22/23 tickers stale (ITC had real data)
    ("AXISBANK.NS",   "2025-03-18"), ("CIPLA.NS",       "2025-03-18"),
    ("DRREDDY.NS",    "2025-03-18"), ("EICHERMOT.NS",   "2025-03-18"),
    ("HCLTECH.NS",    "2025-03-18"), ("HDFCBANK.NS",    "2025-03-18"),
    ("HINDUNILVR.NS", "2025-03-18"), ("ICICIBANK.NS",   "2025-03-18"),
    ("INFY.NS",       "2025-03-18"), ("KOTAKBANK.NS",   "2025-03-18"),
    ("LT.NS",         "2025-03-18"), ("M&M.NS",         "2025-03-18"),
    ("MARUTI.NS",     "2025-03-18"), ("NESTLEIND.NS",   "2025-03-18"),
    ("NTPC.NS",       "2025-03-18"), ("ONGC.NS",        "2025-03-18"),
    ("RELIANCE.NS",   "2025-03-18"), ("SBIN.NS",        "2025-03-18"),
    ("SUNPHARMA.NS",  "2025-03-18"), ("TCS.NS",         "2025-03-18"),
    ("ULTRACEMCO.NS", "2025-03-18"), ("WIPRO.NS",       "2025-03-18"),
}


def main():
    # =================================================================
    # 0. Safety check — remove old DB if it exists (clean migration)
    # =================================================================
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  Removed existing database: %s" % DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # =================================================================
    # 1. Create schema
    # =================================================================
    print("=" * 70)
    print("STEP 1: Creating schema")
    print("=" * 70)

    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    print("  Schema created from %s" % SCHEMA_PATH)

    # =================================================================
    # 2. Insert tickers
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Inserting tickers (%d)" % len(TICKER_META))
    print("=" * 70)

    ticker_ids = {}
    for symbol, (name, sector, exch) in TICKER_META.items():
        notes = None
        replaces = None
        if symbol == "EICHERMOT.NS":
            notes = "Replaced TATAMOTORS.NS (demerger)"
            replaces = "TATAMOTORS.NS"

        cur.execute(
            """INSERT OR IGNORE INTO tickers
               (symbol, company_name, sector, exchange, replaces_symbol, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (symbol, name, sector, exch, replaces, notes)
        )
        cur.execute("SELECT ticker_id FROM tickers WHERE symbol = ?", (symbol,))
        ticker_ids[symbol] = cur.fetchone()[0]
        print("  [%2d] %-16s  %-35s  %s" % (ticker_ids[symbol], symbol, name, sector))

    conn.commit()
    print("  Inserted %d tickers" % len(ticker_ids))

    # =================================================================
    # 3. Load trading calendar
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Loading trading calendar")
    print("=" * 70)

    calendar = pd.read_csv(CALENDAR_PATH, parse_dates=["Date"])
    cal_rows = 0
    for _, row in calendar.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        cur.execute(
            "INSERT OR IGNORE INTO trading_calendar (trade_date) VALUES (?)",
            (date_str,)
        )
        cal_rows += 1

    conn.commit()
    print("  Inserted %d calendar dates" % cal_rows)
    print("  Range: %s to %s" % (
        calendar["Date"].min().strftime("%Y-%m-%d"),
        calendar["Date"].max().strftime("%Y-%m-%d")))

    # =================================================================
    # 4. Insert daily prices
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Loading daily prices")
    print("=" * 70)

    rec = pd.read_pickle(PICKLE_PATH)
    tickers_in_data = sorted(set(c[0] for c in rec.columns))
    print("  Tickers in pickle: %d" % len(tickers_in_data))
    print("  Trading days in pickle: %d" % len(rec))

    rows_inserted = 0
    quality_counts = {"clean": 0, "forward_filled": 0, "vol_missing": 0}

    for t in tickers_in_data:
        df = rec[t]
        tid = ticker_ids[t]
        ticker_rows = 0

        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d")

            # Classify data quality
            if (t, date_str) in KNOWN_FFILLED_ROWS:
                quality = "forward_filled"
            elif pd.isna(row["Volume"]) or row["Volume"] == 0:
                # Check if it's truly stale (flat OHLC + zero vol) or just missing volume
                is_flat = (row["Open"] == row["High"] == row["Low"] == row["Close"])
                if is_flat and (pd.isna(row["Volume"]) or row["Volume"] == 0):
                    quality = "forward_filled"
                else:
                    quality = "vol_missing"
            else:
                quality = "clean"

            volume = None if pd.isna(row["Volume"]) else int(row["Volume"])
            # Treat zero volume as NULL (unknown) unless it's a known forward-fill
            if volume == 0:
                volume = None

            cur.execute(
                """INSERT OR REPLACE INTO daily_prices
                   (ticker_id, trade_date, open, high, low, close, volume, data_quality)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (tid, date_str,
                 float(row["Open"]), float(row["High"]),
                 float(row["Low"]), float(row["Close"]),
                 volume, quality)
            )
            rows_inserted += 1
            ticker_rows += 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        print("  %-16s  ticker_id=%2d  rows=%d" % (t, tid, ticker_rows))

    conn.commit()
    print("\n  Total rows inserted: %d" % rows_inserted)
    print("  Data quality breakdown:")
    for q, c in sorted(quality_counts.items()):
        print("    %-20s %d" % (q, c))

    # =================================================================
    # 5. Log known errata into data_quality_log
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Logging known data quality issues")
    print("=" * 70)

    errata = [
        # NESTLEIND pre-trading placeholders
        ("NESTLEIND.NS", "2010-01-04", "stale_row",
         "Pre-trading placeholder: flat OHLC + zero volume",
         "Forward-filled from first real trading day"),
        ("NESTLEIND.NS", "2010-01-05", "stale_row",
         "Pre-trading placeholder: flat OHLC + zero volume",
         "Forward-filled from first real trading day"),
        ("NESTLEIND.NS", "2010-01-06", "stale_row",
         "Pre-trading placeholder: flat OHLC + zero volume",
         "Forward-filled from first real trading day"),
        ("NESTLEIND.NS", "2010-01-07", "stale_row",
         "Pre-trading placeholder: flat OHLC + zero volume",
         "Forward-filled from first real trading day"),
        # MARUTI single stale row
        ("MARUTI.NS", "2015-08-12", "stale_row",
         "Single stale row: flat OHLC + zero volume",
         "Forward-filled from adjacent trading day"),
        # M&M single stale row
        ("M&M.NS", "2025-09-08", "stale_row",
         "Single stale row: flat OHLC + zero volume",
         "Forward-filled from adjacent trading day"),
    ]

    # Add 2025-03-18 bulk stale event (22/23 tickers; ITC had real data)
    stale_2025_03_18_tickers = [
        "AXISBANK.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
        "HCLTECH.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
        "INFY.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
        "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
        "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS",
        "ULTRACEMCO.NS", "WIPRO.NS",
    ]
    for t in stale_2025_03_18_tickers:
        errata.append((t, "2025-03-18", "stale_row",
                       "Bulk stale event: 22/23 tickers stale (ITC.NS had real data)",
                       "Forward-filled; kept in calendar because ITC had real trading"))

    log_count = 0
    for symbol, date, issue, desc, resolution in errata:
        tid = ticker_ids.get(symbol)
        if tid:
            cur.execute(
                """INSERT INTO data_quality_log
                   (ticker_id, trade_date, issue_type, description, resolution)
                   VALUES (?, ?, ?, ?, ?)""",
                (tid, date, issue, desc, resolution)
            )
            log_count += 1

    conn.commit()
    print("  Logged %d data quality entries" % log_count)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)

    cur.execute("SELECT COUNT(*) FROM tickers")
    print("  tickers:          %d rows" % cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM daily_prices")
    print("  daily_prices:     %d rows" % cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM trading_calendar")
    print("  trading_calendar: %d rows" % cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM data_quality_log")
    print("  data_quality_log: %d rows" % cur.fetchone()[0])
    cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices")
    min_d, max_d = cur.fetchone()
    print("  Date range:       %s to %s" % (min_d, max_d))

    print("\n  Database saved to: %s" % DB_PATH)
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
