# Contributing

Thanks for helping improve **crypto-vol-quickstart**! This guide keeps diffs tidy and the workflow reproducible.

## 1) Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if you have it
pre-commit install
2) Branching & commits
Create feature branches from main (or your current work branch):

feat/onchain-sql, chore/qa-script, fix/hedge-bug

Use Conventional Commits:

feat: add Dune-ready on-chain SQL pack

fix(hedge): guard against missing funding

chore(repo): ignore caches under runs/

docs(README): link on-chain data dictionary

3) Pre-commit & Jupytext
We pair .ipynb with .py and auto-sync via pre-commit. Before pushing:

bash
Copy code
pre-commit run -a
If a hook modifies files (common with jupytext), stage those edits and commit again.

4) Data hygiene (important)
Do not version large data or generated artifacts:

runs/**, data/interim/**, most of data/processed/**

Exceptions: small demo files or docs like data/processed/onchain/README.md.

Keep secrets out of Git. Use .env (gitignored) for keys (e.g., DUNE_API_KEY).

5) Style & formatting
The repo ships with a .editorconfig (indent, newlines, trailing spaces).

Python: 4 spaces, LF newlines, final newline at EOF.

SQL: 2 spaces (compact CTE style).

Makefiles: TABs for recipes.

Markdown: we preserve double-space line breaks.

6) Running QA locally
Quality-check your market data (spot/perp/funding):

bash
Copy code
make qa-data   SPOT=data/raw/binance_spot_BTCUSDT_5m.parquet   PERP=data/raw/binance_perp_BTCUSDT_5m.parquet   FUNDING=data/raw/binance_funding_BTCUSDT.parquet   TF=5min
# view report
open runs/_qa/dq_report.md  # macOS; or cat the file
7) On-chain workflows (optional)
If your Dune/Glassnode access is limited:

Use the Warm cache trick (run with same params in the UI, then fetch via CLI).

Keep SQL under sql/onchain/; avoid duplicates in other directories.

Parameterize start/end/granularity and keep allowlists in CTEs for reproducibility.

8) PR checklist
 pre-commit run -a passes

 No large binaries added; .env not committed

 README/docs updated if behavior or commands changed

 New code has minimal docstrings / comments

 QA report checked when touching data loaders

Happy hacking! 🚀
