"""If pytest is not installed:  python tests/run_all.py"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows terminals (e.g. Git Bash) may not use UTF-8; avoid crashes on symbols such as λ and ±.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failed = total = 0
for mod_name in ["test_data", "test_pca", "test_clustering", "test_backtest"]:
    mod = importlib.import_module(mod_name)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        total += 1
        try:
            getattr(mod, name)()
            print(f"  PASS  {mod_name}.{name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {mod_name}.{name}")
            traceback.print_exc(limit=1)
print(f"\n{total - failed}/{total} tests passed")
sys.exit(1 if failed else 0)
