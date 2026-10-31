"""
Group 6 — "20 stocks, how many bets?"  ·  AD 49F: Data and Decisions

During the presentation each act is run on its own:
    python grup6_demo.py --act 0      # data and cleaning checks
    python grup6_demo.py --act 1      # PCA
    python grup6_demo.py --act 2      # clustering
    python grup6_demo.py --act 3      # walk-forward backtest + absorption ratio
    python grup6_demo.py --act all    # everything (rehearsal)
    python grup6_demo.py --act all --synthetic   # offline rehearsal (synthetic data)
"""
import argparse
import sys

from src import backtest, clustering, data, pca
from src.data import split

# Windows terminals (e.g. Git Bash) may not use UTF-8; avoid crashes on symbols such as λ and ±.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act", default="all", choices=["0", "1", "2", "3", "all"])
    p.add_argument("--synthetic", action="store_true", help="run on synthetic data when there is no internet")
    a = p.parse_args()

    rets, bench, fx = data.load_data(a.synthetic)
    run = (lambda n: a.act in (n, "all"))

    if run("0"):
        data.act0(rets, a.synthetic)
    vecs = None
    if run("1"):
        _, vecs = pca.act1(rets, bench, fx)
    k = None
    if run("2"):
        if vecs is None:
            vecs = pca.fit_pca(split(rets)[0])[1]
        k, _ = clustering.act2(rets, vecs)
    if run("3"):
        backtest.act3(rets, bench, fx, k)


if __name__ == "__main__":
    main()
