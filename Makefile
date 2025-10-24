format:
	black .
	isort .

lint:
	python -m ruff check --fix .
	python -m mypy . || true

test:
	pytest -q

reproduce:
	@echo "Reproduction steps:"
	@echo "1) conda activate quant"
	@echo "2) pip install -r requirements.txt"
	@echo "3) python scripts/test_environment.py"
	@echo "4) python scripts/test_database_system.py"
	@echo "5) python scripts/example_usage.py"


