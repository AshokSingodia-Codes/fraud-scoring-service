.PHONY: setup data train eval test lint serve docker-up load-test clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

data:
	bash scripts/download_data.sh
	python -m src.fraud.data.ingest

train:
	python scripts/train.py

eval:
	python scripts/evaluate.py

test:
	pytest --cov=src/fraud

lint:
	ruff check .

serve:
	uvicorn src.fraud.serving.app:app --reload --port 8000

docker-up:
	docker compose up --build -d

load-test:
	locust -f scripts/locustfile.py --headless -u 10 -r 2 --run-time 1m --host http://localhost:8000

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
