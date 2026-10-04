"""
Diagnostic: check if AXISBANK, HINDUNILVR, LT ever produced BUY signals,
and if so, why they never made it through the top-5 ranking.
"""
from data_access import get_ticker_info
from data_feed import DataFeed
from strategy import MovingAverageCrossoverStrategy, RSIMeanReversionStrategy, Signal

TICKERS_OF_INTEREST = ["AXISBANK.NS", "HINDUNILVR.NS", "LT.NS"]
all_tickers = get_ticker_info()["symbol"].tolist()

feed = DataFeed(all_tickers, "2010-01-04", "2026-08-28")
ma = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
rsi = RSIMeanReversionStrategy(period=14)

LOOKBACK = 50

# Track per-ticker: how many BUY signals each strategy produced
ma_buy_counts = {t: 0 for t in all_tickers}
rsi_buy_counts = {t: 0 for t in all_tickers}

# For the 3 tickers of interest, track their scores on BUY days
# and the competing top-5 cutoff score
interest_ma_buys = {t: [] for t in TICKERS_OF_INTEREST}
interest_rsi_buys = {t: [] for t in TICKERS_OF_INTEREST}
times_in_top5_ma = {t: 0 for t in TICKERS_OF_INTEREST}
times_in_top5_rsi = {t: 0 for t in TICKERS_OF_INTEREST}

bar_count = 0
for date in feed:
    bar_count += 1
    ma_signals = []
    rsi_signals = []
    for ticker in all_tickers:
        bar = feed.current_bar(ticker)
        history = feed.history(ticker, lookback=LOOKBACK)
        ma_sig = ma.generate_signals(ticker, bar, history)
        rsi_sig = rsi.generate_signals(ticker, bar, history)
        ma_signals.append(ma_sig)
        rsi_signals.append(rsi_sig)

        if ma_sig.signal == Signal.BUY:
            ma_buy_counts[ticker] += 1
        if rsi_sig.signal == Signal.BUY:
            rsi_buy_counts[ticker] += 1

    # Check ranking for tickers of interest
    ma_buys_today = sorted([s for s in ma_signals if s.signal == Signal.BUY],
                            key=lambda s: s.score, reverse=True)
    rsi_buys_today = sorted([s for s in rsi_signals if s.signal == Signal.BUY],
                             key=lambda s: s.score, reverse=True)

    ma_top5_tickers = {s.ticker for s in ma_buys_today[:5]}
    rsi_top5_tickers = {s.ticker for s in rsi_buys_today[:5]}

    for t in TICKERS_OF_INTEREST:
        ma_sig = next((s for s in ma_signals if s.ticker == t), None)
        rsi_sig = next((s for s in rsi_signals if s.ticker == t), None)

        if ma_sig and ma_sig.signal == Signal.BUY:
            rank = next((i for i, s in enumerate(ma_buys_today) if s.ticker == t), None)
            interest_ma_buys[t].append({
                "date": str(date.date()), "score": ma_sig.score,
                "rank": rank + 1 if rank is not None else None,
                "total_ma_buys": len(ma_buys_today),
                "top5_cutoff": ma_buys_today[4].score if len(ma_buys_today) >= 5 else None,
            })
            if t in ma_top5_tickers:
                times_in_top5_ma[t] += 1

        if rsi_sig and rsi_sig.signal == Signal.BUY:
            rank = next((i for i, s in enumerate(rsi_buys_today) if s.ticker == t), None)
            interest_rsi_buys[t].append({
                "date": str(date.date()), "score": rsi_sig.score,
                "rank": rank + 1 if rank is not None else None,
                "total_rsi_buys": len(rsi_buys_today),
                "top5_cutoff": rsi_buys_today[4].score if len(rsi_buys_today) >= 5 else None,
            })
            if t in rsi_top5_tickers:
                times_in_top5_rsi[t] += 1

    if bar_count % 500 == 0:
        print("  processed %d / 4109 bars..." % bar_count)

print("\n" + "=" * 70)
print("SIGNAL COUNTS (all %d bars)" % bar_count)
print("=" * 70)
print("\n%-16s  %6s  %6s" % ("Ticker", "MA_BUY", "RSI_BUY"))
for t in sorted(all_tickers):
    marker = " <---" if t in TICKERS_OF_INTEREST else ""
    print("%-16s  %6d  %6d%s" % (t, ma_buy_counts[t], rsi_buy_counts[t], marker))

print("\n" + "=" * 70)
print("TICKERS OF INTEREST — RANKING ANALYSIS")
print("=" * 70)
for t in TICKERS_OF_INTEREST:
    print("\n--- %s ---" % t)
    print("  MA BUY signals: %d" % len(interest_ma_buys[t]))
    print("  RSI BUY signals: %d" % len(interest_rsi_buys[t]))
    print("  Times in MA top-5: %d" % times_in_top5_ma[t])
    print("  Times in RSI top-5: %d" % times_in_top5_rsi[t])

    if interest_ma_buys[t]:
        scores = [x["score"] for x in interest_ma_buys[t]]
        print("  MA score range: %.6f to %.6f (mean %.6f)" % (min(scores), max(scores), sum(scores)/len(scores)))
        # Show a few examples where they were close to top-5
        ranked_examples = [x for x in interest_ma_buys[t] if x["rank"] is not None]
        ranked_examples.sort(key=lambda x: x["rank"])
        print("  Best MA ranks (first 5):")
        for ex in ranked_examples[:5]:
            print("    %s: rank %d/%d, score=%.6f, top5_cutoff=%s" % (
                ex["date"], ex["rank"], ex["total_ma_buys"], ex["score"],
                "%.6f" % ex["top5_cutoff"] if ex["top5_cutoff"] is not None else "N/A"))

    if interest_rsi_buys[t]:
        scores = [x["score"] for x in interest_rsi_buys[t]]
        print("  RSI score range: %.6f to %.6f (mean %.6f)" % (min(scores), max(scores), sum(scores)/len(scores)))
        ranked_examples = [x for x in interest_rsi_buys[t] if x["rank"] is not None]
        ranked_examples.sort(key=lambda x: x["rank"])
        print("  Best RSI ranks (first 5):")
        for ex in ranked_examples[:5]:
            print("    %s: rank %d/%d, score=%.6f, top5_cutoff=%s" % (
                ex["date"], ex["rank"], ex["total_rsi_buys"], ex["score"],
                "%.6f" % ex["top5_cutoff"] if ex["top5_cutoff"] is not None else "N/A"))

print("\nDONE")
