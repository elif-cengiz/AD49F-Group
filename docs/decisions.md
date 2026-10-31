# Honesty contract — fixed decisions

This file is filled in and committed BEFORE looking at any results. Every later change is added as a
separate row with its reason and date (old rows are never deleted).

| Decision | Value | Reason | Date | Decided by |
| --- | --- | --- | --- | --- |
| Data period | 2024-01-01 – 2026-10-01 | Project scope | | |
| Train/test split | 2025-06-30 | Roughly 55% / 45% | | |
| Return type | Daily log returns, auto_adjust=True | Adjusts for bonus issues and dividends | | |
| Missing data rule | Stocks with more than 5% missing data are dropped | | | |
| PCA input | Correlation matrix (standardized returns) | Removes scale differences | | |
| Number of factors | Components above the Marchenko–Pastur threshold | Separates signal from noise | | |
| Clustering | Average linkage, distance sqrt(0.5(1-ρ)) | Mantegna (1999) | | |
| Choice of k | Silhouette, k = 2..8, training data only | | | |
| Backtest | Monthly rebalancing, 250-day look-back, 10 bp one-way cost | | | |

## Later changes

| Date | Change | Reason | Had results been seen? |
| --- | --- | --- | --- |
