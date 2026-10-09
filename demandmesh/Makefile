# Convenience targets (Linux/macOS). On Windows run the python commands directly.
.PHONY: install pipeline tune serve test lint drift-demo docker-build docker-up clean

install:
	pip install -r requirements-dev.txt
	pip install -e .

pipeline:          ## generate data -> clean -> train -> evaluate -> report
	python -m demandmesh.pipeline

tune:              ## same, with randomised hyper-parameter search
	python -m demandmesh.pipeline --tune --n-trials 12

serve:             ## API with hot reload on http://localhost:8000/docs
	uvicorn demandmesh.app:app --reload

test:
	pytest -q

lint:
	ruff check src tests

drift-demo:        ## inflate the last 28 days by 40% and watch PSI alert
	python -m demandmesh.drift --shock 1.4

docker-build:
	docker build --target runtime -t demandmesh-api:1.0.0 .

docker-up:
	docker compose up --build

clean:
	rm -rf data/raw/*.csv* data/processed/*.csv* data/processed/*.json models/v* models/LATEST
