.PHONY: nb lab nb-headless verify-parquet clean-lab which-jupyter lab-install deps

# Prefer venv python; fall back to system python3
VENV        ?= .venv
PY          := $(VENV)/bin/python
ifeq (,$(wildcard $(PY)))
PY          := python3
endif

# Knobs you can override: PORT=8890 NO_BROWSER=1 ROOT_DIR=...
PORT        ?= 8888
NO_BROWSER  ?= 0
ROOT_DIR    ?= $(CURDIR)

# Jupyter Server flags (Notebook 7 / Lab share ServerApp)
LAB_FLAGS = --ServerApp.root_dir="$(ROOT_DIR)" --ServerApp.port=$(PORT) --ServerApp.port_retries=50 --ServerApp.default_url=/lab
ifeq ($(NO_BROWSER),1)
  LAB_FLAGS += --no-browser
endif

# Always launch JupyterLab (nb is alias to lab)
nb: lab

lab:
	@PROJECT_ROOT="$(ROOT_DIR)" PYTHONPATH="$(ROOT_DIR)/src:$$PYTHONPATH" $(PY) -m jupyterlab $(LAB_FLAGS)

# Headless Lab on a custom port (great for SSH/tmux)
nb-headless:
	@$(MAKE) nb NO_BROWSER=1 PORT=8890

# Install/upgrade modern stack into venv
lab-install:
	@$(PY) -m pip install -U "notebook>=7" jupyterlab ipywidgets jupyterlab_widgets

# Install deps from requirements.txt
deps:
	@$(PY) -m pip install -r requirements.txt

# Quick health check: confirm Parquet reading works (no heredoc)
verify-parquet:
	@$(PY) -c "import pandas as pd; from pathlib import Path; p=Path('data/raw/binance_spot_BTCUSDT_5m.parquet'); assert p.exists(), f'Missing file: {p}'; df=pd.read_parquet(p, engine='pyarrow'); print('OK ->', df.shape)"

# Clean JupyterLab caches (if UI looks weird after upgrades)
clean-lab:
	@$(PY) -m jupyter lab clean

# Debug: confirm you're using the venv + versions
which-jupyter:
	@echo "python: $(PY)"
	@$(PY) -c "import sys; print('python', sys.version)"
	@$(PY) -c "import jupyterlab; print('jupyterlab', jupyterlab.__version__)"
	@$(PY) -c "import notebook; print('notebook', notebook.__version__)"

# ---- On-chain MV features (robust; skips missing inputs) ----
.PHONY: onchain-mv
START ?= 2023-01-01
END   ?= 2025-01-01
OUT   ?= data/processed/onchain/mv_feature_set_hourly.parquet

onchain-mv:
	@python scripts/build_onchain_mv.py --start '$(START)' --end '$(END)' --out '$(OUT)'


# ---------- Dune → Parquet exports ----------
DUNE_START ?= 2023-01-01
DUNE_END   ?= 2025-01-01
PERF       ?= large

# Fill these with your saved query IDs from Dune
QID_TRANSFERS      ?= 5876393
QID_UNI_SWAPS      ?= 5876396
QID_ETH_BLOCKS     ?= 5876400
QID_FIRST_SEEN     ?= 5876406

.PHONY: dune-transfers dune-swaps dune-blocks dune-first-seen dune-onchain-all

dune-transfers:
	@python scripts/dune_to_parquet.py \
	  --query-id $(QID_TRANSFERS) \
	  --out data/processed/onchain/transfers.parquet \
	  --param start=$(DUNE_START) --param end=$(DUNE_END) \
	  --performance $(PERF)

dune-swaps:
	@python scripts/dune_to_parquet.py \
	  --query-id $(QID_UNI_SWAPS) \
	  --out data/processed/onchain/uniswap_swaps.parquet \
	  --param start=$(DUNE_START) --param end=$(DUNE_END) \
	  --performance $(PERF)

dune-blocks:
	@python scripts/dune_to_parquet.py \
	  --query-id $(QID_ETH_BLOCKS) \
	  --out data/processed/onchain/eth_blocks.parquet \
	  --param start=$(DUNE_START) --param end=$(DUNE_END) \
	  --performance $(PERF)

dune-first-seen:
	@python scripts/dune_to_parquet.py \
	  --query-id $(QID_FIRST_SEEN) \
	  --out data/processed/onchain/address_first_seen.parquet \
	  --param start=$(DUNE_START) --param end=$(DUNE_END) \
	  --performance $(PERF)

# Fetch all four in one go
dune-onchain-all: dune-transfers dune-swaps dune-blocks dune-first-seen
## === On-Chain Queries ===
.PHONY: onchain-tree onchain-queries onchain-readme onchain-commit

# Show where queries live
onchain-tree:
	@echo "analytics/onchain/queries:"
	@find analytics/onchain/queries -maxdepth 1 -type f -name "*.sql" -print

# Ensure queries exist (no-op here since files are versioned)
onchain-queries: onchain-tree
	@echo "Queries are managed in git under analytics/onchain/queries/"

# Re-append the README appendix if missing (guarded by marker)
onchain-readme:
	@grep -q "<!-- ONCHAIN-APPENDIX:BEGIN -->" README.md && echo "Appendix present ✅" || (echo "Appendix missing, re-run the CLI block from ChatGPT to append it." && exit 1)

# Commit just the on-chain assets
onchain-commit:
	git add analytics/onchain/queries README.md Makefile
	git commit -m "Add on-chain queries (transfers, transfers_cex_flags, uniswap_swaps, eth_blocks, address_first) + README appendix + Makefile targets" || echo "Nothing to commit"

## === Dune Exports ===
.NOTPARALLEL:

.PHONY: dune-install dune-onchain-all dune-onchain-one dune-onchain-check dune-env
dune-install:
	@python -m pip install -r requirements.txt

# Load .env if present (portable; no sed)
dune-env:
	@/bin/sh -c '[ -f .env ] && . ./.env; true'

# Run all jobs sequentially via saved query IDs (optional SLEEP between jobs)
# Example:
#   make dune-onchain-all START=2025-09-01T00:00:00Z END=2025-09-01T06:00:00Z SLEEP=2
dune-onchain-all: dune-env
	@python -m src.onchain.dune_export --jobs analytics/onchain/config/onchain_jobs.yaml --outdir data/processed/onchain $(if $(START),--start $(START),) $(if $(END),--end $(END),) $(if $(SLEEP),--sleep $(SLEEP),)

# Run a single job:
#   make dune-onchain-one JOB=eth_blocks START=... END=...
dune-onchain-one: dune-env
	@test -n "$(JOB)" || (echo "Set JOB=<name> (e.g., eth_blocks)" && exit 2)
	@python -m src.onchain.dune_export --jobs analytics/onchain/config/onchain_jobs.yaml --outdir data/processed/onchain --job $(JOB) $(if $(START),--start $(START),) $(if $(END),--end $(END),)

# YAML sanity
dune-onchain-check:
	@python -c "import yaml; yaml.safe_load(open('analytics/onchain/config/onchain_jobs.yaml')); print('YAML OK ✅')"
