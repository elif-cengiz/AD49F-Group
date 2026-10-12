# AD49F — Group 6: Unsupervised ML1 Applications

**"20 stocks, how many bets?"** — Hidden risk factors in BIST equities with PCA and clustering

AD 49F: Data and Decisions — Building Your Own Investing Algorithms with AI · Boğaziçi University

## Team

| Member | On stage | Responsibility | Code |
| --- | --- | --- | --- |
| Elif Sude Cengiz (A) | Opening, Act 1: PCA, Q&A moderator | Story + PCA lead; demo script | `src/config.py`, `src/pca.py`, `grup6_demo.py`, `tests/helpers.py` |
| Ahmet Çevik (B) | Act 2: Clustering, Closing | Clustering + live Claude Code demo; slides | `src/plotting.py`, `src/clustering.py`, `tests/run_all.py` |
| Esat Çankaya (C) | Data section, Act 3: Honest test | Data pipeline, honest backtest, honesty contract | `src/data.py`, `src/backtest.py`, `docs/decisions.md` |

Team: Elif Sude Cengiz, Esat Çankaya, Ahmet Çevik.

## Hypothesis

An investor holding 20 liquid BIST stocks believes they are making 20 independent bets. PCA and clustering show
that the real number of independent risk sources is much smaller, and that it shrinks further in stress periods.
We turn this into a portfolio decision: a cluster-balanced portfolio versus an equal-weight portfolio, compared
in a leakage-free walk-forward backtest.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
```

## Running

```bash
python grup6_demo.py --act 0       # data and cleaning checks
python grup6_demo.py --act 1       # PCA
python grup6_demo.py --act 2       # clustering
python grup6_demo.py --act 3       # walk-forward backtest + absorption ratio
python grup6_demo.py --act all --synthetic    # offline rehearsal with synthetic data
python tests/run_all.py            # or: pytest tests/
```

The first run saves the data to `data/`; later runs work offline.
Charts are written to `figs/`.

## Project structure

```text
grup6_demo.py        entry point run during the presentation
src/config.py        dates, tickers, fixed parameters (honesty contract)
src/data.py          download, cache, cleaning (holiday rows, ±10% check)
src/pca.py           PCA, Marchenko–Pastur noise threshold, effective number of bets
src/clustering.py    correlation distance, hierarchical clustering, choice of k, stability
src/backtest.py      walk-forward backtest, attribution, absorption ratio
src/plotting.py      shared chart style
tests/               23 sanity tests
docs/                decisions, results, AI usage log
```

## Honesty contract

Parameters were fixed in `src/config.py` and `docs/decisions.md` before looking at results:
train/test split on 2025-06-30, 250-day look-back window, monthly rebalancing, 10 bp one-way transaction cost,
and k chosen on training data only.

## Use of AI

Part of the code was written with Claude; every piece is checked by the tests in `tests/`.
Prompts, outputs and the bugs we caught are logged in `docs/ai_log.md`.
