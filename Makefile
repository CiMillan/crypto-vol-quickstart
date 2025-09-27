# -------- Makefile (tabs required) --------
.PHONY: venv deps kernel nb nb-nosrc clean

venv:
	@if [ ! -d .venv ]; then \
		echo "→ Creating .venv"; \
		python3 -m venv .venv; \
	fi
	@. ./.venv/bin/activate && python -V

deps: venv
	@. ./.venv/bin/activate && pip install --upgrade pip && \
	( pip install -r requirements.txt ) || pip install numpy pandas matplotlib ccxt statsmodels ipykernel notebook

kernel: venv
	@. ./.venv/bin/activate && python -m ipykernel install --user --name=crypto-vol --display-name "crypto-vol (.venv)"

nb: venv
	@. ./.venv/bin/activate && python -m notebook

nb-nosrc: venv
	@./.venv/bin/python -m notebook

clean:
	@jupyter kernelspec remove -y crypto-vol || true
# -------------------------------------------
