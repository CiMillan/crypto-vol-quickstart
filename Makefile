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
