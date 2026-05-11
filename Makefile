.PHONY: install download-weights test run-basichouse run-synagoge clean docker-build docker-run-basichouse docker-run-synagoge

# -----------------------------------------------------------------------------
# Local Development
# -----------------------------------------------------------------------------

install:
	@echo "Installing dependencies using uv..."
	uv venv --python 3.11 .venv
	. .venv/bin/activate && uv pip install -e .
	. .venv/bin/activate && uv pip install -r requirements.txt
	@echo "Installation complete. Activate the virtual environment with: source .venv/bin/activate"

download-weights:
	@echo "Downloading model weights..."
	.venv/bin/python scripts/download_weights.py

test:
	@echo "Running tests..."
	.venv/bin/python -m pytest tests/ -v

run-basichouse:
	@echo "Running pipeline on BasicHouse (local)..."
	.venv/bin/python scripts/run_pipeline.py \
		--scene data/BasicHouse_with_pc \
		--device cuda \
		--keyframes 0 40 80 120 159 \
		--max_per_frame 8 \
		--out outputs/basichouse

run-synagoge:
	@echo "Running pipeline on synagoge (local)..."
	.venv/bin/python scripts/run_pipeline.py \
		--scene data/synagoge_with_pc \
		--device cuda \
		--auto_keyframes 8 \
		--out outputs/synagoge

clean:
	@echo "Cleaning up generated files and caches..."
	rm -rf __pycache__ .pytest_cache
	find src scripts tests -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf outputs/*
	@echo "Clean complete."

# -----------------------------------------------------------------------------
# Docker Operations
# -----------------------------------------------------------------------------

DOCKER_IMAGE = rgbdsg:latest

docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE)..."
	docker build -t $(DOCKER_IMAGE) .

docker-run-basichouse:
	@echo "Running pipeline on BasicHouse (Docker)..."
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/weights:/app/weights \
		$(DOCKER_IMAGE) \
		python scripts/run_pipeline.py \
			--scene data/BasicHouse_with_pc \
			--device cpu \
			--keyframes 0 40 80 120 159 \
			--max_per_frame 8 \
			--out outputs/basichouse

docker-run-synagoge:
	@echo "Running pipeline on synagoge (Docker)..."
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/weights:/app/weights \
		$(DOCKER_IMAGE) \
		python scripts/run_pipeline.py \
			--scene data/synagoge_with_pc \
			--device cpu \
			--auto_keyframes 8 \
			--out outputs/synagoge
