# Results (real data)

Data download date: ____  ·  Commit: ____  ·  This file is frozen after `git tag v1-results`.

## Act 0 — Data (Member C)
- Number of stocks / trading days: ____ / ____
- Train / test days: ____ / ____
- Days beyond ±10% in unadjusted prices, and the most striking example: ____

## Act 1 — PCA (Member A)
- PC1 variance share: 49.3% · First 3 PCs: 62.7%
- Marchenko–Pastur λ+: 1.52 · Components above noise: 2 (the 70% rule would keep 5)
- Effective number of bets: 8.0 / 20
- Correlations of PCs with XU100 / USD/TRY: PC1 0.98 / −0.10 · PC2 0.04 / 0.19 · PC3 0.03 / −0.03
- Do Spearman / Ledoit–Wolf change the result? No: effective bets 8.9 (Spearman) and 8.4 (Ledoit–Wolf). Leakage check: PC2 loadings change by up to 0.22 if the test period is included.

## Act 2 — Clustering (Member B)
- Chosen k and silhouette scores: ____
- Cluster members: ____
- Train–test ARI (stability): ____ · ARI with sectors: ____
- Surprising stock(s): ____

## Act 3 — Test (Member C)
- Metrics table (TRY and USD): ____
- Leakage demo (honest vs. leaky return/vol): ____
- Period(s) with the highest absorption ratio: ____

## Opening poll
- The class's guess: ____ · Actual effective number of bets: ____
