# --- Configuration Variables ---
IP_ADDRESS := $(shell hostname -I | awk '{print $$1}')

# The port you want to access Jupyter on
PORT := 9999

# The security token for login (leave empty to generate random)
TOKEN := mysecret

# Python Interpreter (Change to 'python3' if needed)
PYTHON := python

# --- Commands ---

.PHONY: help install run clean

default: run

help:
	@echo "Available commands:"
	@echo "  make install  - Install JupyterLab globally (System scope)"
	@echo "  make run      - Start JupyterLab on port $(PORT) without browser"
	@echo "  make clean    - Remove pycache and checkpoints"

install:
	@echo "Installing JupyterLab to global environment..."
	$(PYTHON) -m pip install jupyterlab numpy matplotlib scipy scikit-learn pytorch torchvision

run:
	@echo "Starting JupyterLab on port $(PORT)..."
	@echo "Access URL: http://$(IP_ADDRESS):$(PORT)/lab?token=$(TOKEN)"
	@# Flags explanation:
	@# --no-browser: Prevent auto-opening the browser
	@# --port: Specify the port
	@# --ip: Bind to $(IP_ADDRESS) for security
	@# --notebook-dir: Force root to current directory
	@# --NotebookApp.token: Hardcode token for easy access
	@# --ServerApp.iopub_msg_rate_limit / --ServerApp.rate_limit_window: increase output rate limits
	jupyter lab \
		--no-browser \
		--port=$(PORT) \
		--ip=0.0.0.0 \
		--notebook-dir="." \
		--ServerApp.iopub_msg_rate_limit=100000 \
		--ServerApp.rate_limit_window=10 \
		--NotebookApp.iopub_msg_rate_limit=100000 \
		--NotebookApp.rate_limit_window=10 \
		--NotebookApp.token='$(TOKEN)' \
		--NotebookApp.password='' \
		--allow-root

clean:
	@echo "Cleaning up runtime artifacts..."
	-rm -rf .ipynb_checkpoints
	-rm -rf **/ .ipynb_checkpoints
	-rm -rf __pycache__
	-rm -rf **/__pycache__

sh:
	docker compose exec app bash