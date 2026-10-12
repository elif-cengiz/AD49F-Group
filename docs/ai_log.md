# Claude Code usage log

One entry per meaningful session. The "we audited the tool" moment in the presentation is chosen from here.

## Template

### [Date] — [Member] — [Topic]
- **Prompt:** (copy it exactly)
- **What Claude produced:** (file/function name, commit hash)
- **How we verified it:** (which test, which manual check)
- **Bug caught / fix:** (if any; screenshot: docs/img/...)
- **Lesson:** (one sentence)

## Entries

### 2026-10-07 — Preparation — Benchmark alignment bug
- **Prompt:** "Write a sanity test checking that the correlation between PC scores and XU100 is above 0.8."
- **What Claude produced:** tests/test_pca.py::test_pc1_is_market_factor
- **How we verified it:** The test FAILED on its first run. Cause: the XU100 return was differenced after the
  prices were aligned to the return dates, so the first day stayed NaN and np.corrcoef returned NaN.
  pandas .corr() silently skips NaN, so the demo output looked fine.
- **Bug caught / fix:** Align to the BIST calendar first, then take differences (src/data.py, load_data).
- **Lesson:** Silent NaNs are the most dangerous bugs; output that "looks right" is not enough.

### 2026-10-09 — Elif — First run on real data: two data problems
- **Prompt:** "Here is the output of `python grup6_demo.py --act all` on real data. What looks wrong?"
- **What Claude produced:** `outside_price_limit()` and `_download_close()` in src/data.py; a new test in tests/test_data.py
- **How we verified it:** ACT 0 reported 24 "impossible" days in ADJUSTED returns (should be 0). All were about -0.105,
  e.g. 19–21 March 2025 (a market-wide sell-off with many stocks limit-down). A -10% move is log(0.9) = -0.105 in log
  returns, but the check used |r| > log(1.11) = 0.104, so genuine limit-down days were flagged. Separately, ASELS failed
  to download ("database is locked", a yfinance threading issue) and was silently dropped, leaving 19 stocks.
- **Bug caught / fix:** Asymmetric limits in log space [log(0.89), log(1.11)]; download without threads and retry
  empty tickers one by one. New test: a -10% day is not flagged, a halving price is.
- **After the fix:** 0 days beyond the limit in both adjusted and "unadjusted" prices. Yahoo's unadjusted close is
  already adjusted for bonus issues, so the bonus-issue trap does not appear in Yahoo data (it does in raw exchange data).
- **Lesson:** A ±10% rule is not ±0.10 in log returns. And a warning that drops a stock changes every result after it.
