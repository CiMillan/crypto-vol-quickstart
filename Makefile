SHELL := /bin/bash
.ONESHELL:

PY := $(CURDIR)/.venv/bin/python
PIP := $(CURDIR)/.venv/bin/pip

# ====== Defaults ======
SYMBOL_SPOT ?= BTC/USDT
SYMBOL_PERP ?= BTCUSDT
HORIZON     ?= 12

# Multi-asset list for *-all targets (perp symbols)
ASSETS ?= BTCUSDT ETHUSDT

# ---------- QUICK (fast dev) ----------
QUICK_SINCE      ?= 2024-01-01
QUICK_UNTIL      ?= 2025-09-20
QUICK_TIMEFRAME  ?= 5m
QUICK_SPOT_FILE      := data/raw/binance_spot_$(subst /,,$(SYMBOL_SPOT))_$(QUICK_TIMEFRAME).parquet
QUICK_FUNDING_FILE   := data/raw/binance_funding_$(SYMBOL_PERP).parquet
QUICK_FEATURES_FILE  := data/processed/$(SYMBOL_PERP)_$(QUICK_TIMEFRAME).parquet
QUICK_RUN_DIR        := runs/$(SYMBOL_PERP)

# ---------- FULL (heavy) ----------
FULL_SINCE       ?= 2020-01-01
FULL_UNTIL       ?= 2025-09-20
FULL_TIMEFRAME   ?= 1m
FULL_SPOT_FILE       := data/raw/binance_spot_$(subst /,,$(SYMBOL_SPOT))_$(FULL_TIMEFRAME).parquet
FULL_FUNDING_FILE    := data/raw/binance_funding_$(SYMBOL_PERP)_full.parquet
FULL_FEATURES_FILE   := data/processed/$(SYMBOL_PERP)_$(FULL_TIMEFRAME).parquet
FULL_RUN_DIR         := runs/$(SYMBOL_PERP)_full

.PHONY: venv quick quick-data quick-features quick-exp quick-backtest \
        full full-data full-features full-exp full-backtest \
        quick-all full-all clean clean-data clean-runs clean-pyc help


venv:
	$(shell command -v python3.11 || command -v python3) -m venv .venv || true
	$(PIP) install --upgrade pip
	# Torch optional on macOS; failing is fine
	$(PIP) install -r requirements.txt

# ===== QUICK single-asset =====
quick: quick-data quick-features quick-exp quick-backtest

quick-data: venv
	$(PY) -m src.data.ccxt_download --symbol $(SYMBOL_SPOT) \
		--timeframe $(QUICK_TIMEFRAME) --since $(QUICK_SINCE) --until $(QUICK_UNTIL) \
		--limit 1000 --out $(QUICK_SPOT_FILE)
	$(PY) -m src.data.binance_funding --symbol $(SYMBOL_PERP) \
		--start $(QUICK_SINCE) --end $(QUICK_UNTIL) --out $(QUICK_FUNDING_FILE)

quick-features: venv
	$(PY) -m src.features.make_features --spot $(QUICK_SPOT_FILE) \
		--funding $(QUICK_FUNDING_FILE) --symbol $(SYMBOL_PERP) \
		--timeframe $(QUICK_TIMEFRAME) --out $(QUICK_FEATURES_FILE)

quick-exp: venv
	$(PY) -m src.modeling.run_experiments --data $(QUICK_FEATURES_FILE) \
		--symbol $(SYMBOL_PERP) --horizon $(HORIZON) --output $(QUICK_RUN_DIR)

quick-backtest: venv
	$(PY) -m src.backtest.vol_targeting --pred $(QUICK_RUN_DIR)/predictions.parquet \
		--retcol ret_5 --fee_bps 1 --output $(QUICK_RUN_DIR)/backtest_$(SYMBOL_PERP).json

# ===== FULL single-asset =====
full: full-data full-features full-exp full-backtest

full-data: venv
	$(PY) -m src.data.ccxt_download --symbol $(SYMBOL_SPOT) \
		--timeframe $(FULL_TIMEFRAME) --since $(FULL_SINCE) --until $(FULL_UNTIL) \
		--limit 1000 --out $(FULL_SPOT_FILE)
	$(PY) -m src.data.binance_funding --symbol $(SYMBOL_PERP) \
		--start $(FULL_SINCE) --end $(FULL_UNTIL) --out $(FULL_FUNDING_FILE)

full-features: venv
	$(PY) -m src.features.make_features --spot $(FULL_SPOT_FILE) \
		--funding $(FULL_FUNDING_FILE) --symbol $(SYMBOL_PERP) \
		--timeframe $(FULL_TIMEFRAME) --out $(FULL_FEATURES_FILE)

full-exp: venv
	$(PY) -m src.modeling.run_experiments --data $(FULL_FEATURES_FILE) \
		--symbol $(SYMBOL_PERP) --horizon $(HORIZON) --output $(FULL_RUN_DIR)

full-backtest: venv
	$(PY) -m src.backtest.vol_targeting --pred $(FULL_RUN_DIR)/predictions.parquet \
		--retcol ret_5 --fee_bps 1 --output $(FULL_RUN_DIR)/backtest_$(SYMBOL_PERP).json

# ===== QUICK multi-asset (BTC+ETH by default) =====
quick-all: venv
	for a in $(ASSETS); do \
		SPOT="$${a%USDT}/USDT"; \
		SPOTFILE="data/raw/binance_spot_$${a}_$(QUICK_TIMEFRAME).parquet"; \
		FUNDFILE="data/raw/binance_funding_$${a}.parquet"; \
		FEATFILE="data/processed/$${a}_$(QUICK_TIMEFRAME).parquet"; \
		RUNDIR="runs/$${a}"; \
		echo "=== QUICK $$a ==="; \
		$(PY) -m src.data.ccxt_download --symbol $$SPOT \
			--timeframe $(QUICK_TIMEFRAME) --since $(QUICK_SINCE) --until $(QUICK_UNTIL) \
			--limit 1000 --out $$SPOTFILE; \
		$(PY) -m src.data.binance_funding --symbol $$a \
			--start $(QUICK_SINCE) --end $(QUICK_UNTIL) --out $$FUNDFILE; \
		$(PY) -m src.features.make_features --spot $$SPOTFILE \
			--funding $$FUNDFILE --symbol $$a --timeframe $(QUICK_TIMEFRAME) \
			--out $$FEATFILE; \
		$(PY) -m src.modeling.run_experiments --data $$FEATFILE \
			--symbol $$a --horizon $(HORIZON) --output $$RUNDIR; \
		$(PY) -m src.backtest.vol_targeting --pred $$RUNDIR/predictions.parquet \
			--retcol ret_5 --fee_bps 1 --output $$RUNDIR/backtest_$${a}.json; \
	done

# ===== FULL multi-asset =====
full-all: venv
	for a in $(ASSETS); do \
		SPOT="$${a%USDT}/USDT"; \
		SPOTFILE="data/raw/binance_spot_$${a}_$(FULL_TIMEFRAME).parquet"; \
		FUNDFILE="data/raw/binance_funding_$${a}_full.parquet"; \
		FEATFILE="data/processed/$${a}_$(FULL_TIMEFRAME).parquet"; \
		RUNDIR="runs/$${a}_full"; \
		echo "=== FULL $$a ==="; \
		$(PY) -m src.data.ccxt_download --symbol $$SPOT \
			--timeframe $(FULL_TIMEFRAME) --since $(FULL_SINCE) --until $(FULL_UNTIL) \
			--limit 1000 --out $$SPOTFILE; \
		$(PY) -m src.data.binance_funding --symbol $$a \
			--start $(FULL_SINCE) --end $(FULL_UNTIL) --out $$FUNDFILE; \
		$(PY) -m src.features.make_features --spot $$SPOTFILE \
			--funding $$FUNDFILE --symbol $$a --timeframe $(FULL_TIMEFRAME) \
			--out $$FEATFILE; \
		$(PY) -m src.modeling.run_experiments --data $$FEATFILE \
			--symbol $$a --horizon $(HORIZON) --output $$RUNDIR; \
		$(PY) -m src.backtest.vol_targeting --pred $$RUNDIR/predictions.parquet \
			--retcol ret_5 --fee_bps 1 --output $$RUNDIR/backtest_$${a}.json; \
	done

# ===== Cleaning =====
clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +; \
	find . -name '*.pyc' -o -name '*.pyo' -delete

clean-data:
	rm -rf data/raw data/processed
	mkdir -p data/raw data/processed
	touch data/raw/.gitkeep data/processed/.gitkeep

clean-runs:
	rm -rf runs
	mkdir -p runs
	touch runs/.gitkeep

clean: clean-pyc clean-data clean-runs

help:
	@echo "Targets:"
	@echo "  make quick            # single-asset fast run (BTC by default)"
	@echo "  make full             # single-asset heavy run"
	@echo "  make quick-all        # multi-asset fast run (ASSETS=$(ASSETS))"
	@echo "  make full-all         # multi-asset heavy run"
	@echo "  make clean            # remove data/processed, data/raw, runs and pyc caches"
	@echo "Vars you can override: SYMBOL_SPOT, SYMBOL_PERP, ASSETS, HORIZON, *_SINCE/*_UNTIL, *_TIMEFRAME"
.PHONY: eda-init
eda-init:
	# EDA folders
	mkdir -p notebooks/eda/{00_overview,01_data_quality,02_returns_volatility,03_features,04_leakage_checks,05_visualizations,_templates}
	# Data + reports
	mkdir -p data/{raw,interim,processed,external}
	mkdir -p reports/{figures,tables}
	# Utils + configs + tests
	mkdir -p src/eda_utils
	mkdir -p configs/{notebooks,plots}
	mkdir -p tests/eda
	# README scaffold
	test -f notebooks/eda/README.md || printf "## EDA Notebooks\n\nUse numbered subfolders to keep order.\n" > notebooks/eda/README.md
	@echo "EDA folders created ✅"
.PHONY: eda-kernel
eda-kernel: venv
	$(PY) -m pip install --upgrade pip ipykernel
	$(PY) -m ipykernel install --user --name crypto-vol-quickstart-311 --display-name "Python 3.11 (.venv)"
	@mkdir -p .vscode
	@printf '%s\n' '{' \
'"python.defaultInterpreterPath": ".venv/bin/python",' \
'"jupyter.jupyterServerType": "local",' \
'"jupyter.kernelPickerType": "Insiders",' \
'"jupyter.kernels.filter": [{"path": "crypto-vol-quickstart-311", "include": true}]' \
'}' > .vscode/settings.json
	@echo "Jupyter kernel + VS Code settings ready ✅"
.PHONY: eda-jupytext-init eda-new eda-sync
eda-jupytext-init: venv
	$(PIP) install jupytext jupyter
	@test -f jupytext.toml || printf "formats = \"ipynb,py:percent\"\n" > jupytext.toml
	@echo "Jupytext configured ✅"

# Create a new paired notebook: make eda-new NAME=my_notebook FOLDER=01_data_quality
eda-new:
	@test -n "$(NAME)" || (echo "Usage: make eda-new NAME=my_notebook [FOLDER=00_overview]"; exit 1)
	@f=$(if $(FOLDER),$(FOLDER),00_overview); \
	mkdir -p notebooks/eda/$$f; \
	path_py="notebooks/eda/$$f/$(NAME).py"; \
	test -f $$path_py || cp notebooks/eda/_templates/eda_starter.py $$path_py; \
	$(PY) -m pip install jupytext >/dev/null 2>&1 || true; \
	jupytext --set-formats ipynb,py:percent $$path_py; \
	jupytext --sync $$path_py; \
	echo "Created & paired: $$path_py ↔ $${path_py%.py}.ipynb ✅"

# Sync all paired notebooks in notebooks/eda
eda-sync:
	$(PY) -m pip install jupytext >/dev/null 2>&1 || true
	find notebooks/eda -type f \( -name "*.ipynb" -o -name "*.py" \) -print0 | xargs -0 -n 50 jupytext --sync
	@echo "Synced all paired notebooks ✅"
