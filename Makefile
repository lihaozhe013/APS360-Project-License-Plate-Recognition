# --- Configuration Variables ---
IP_ADDRESS := $(shell hostname -I | awk '{print $$1}')

# The port you want to access Jupyter on
PORT := 9999

# The security token for login (leave empty to generate random)
TOKEN := mysecret

.PHONY: help install run clean

default: run

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
	uv run jupyter lab \
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

data:
	cd data_generation && uv run pipeline.py

train-crnn:
	cd train/src && uv run train.py

test:
	cd train/src && uv run inference.py