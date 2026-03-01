.PHONY: up down api frontend test lint reproduce clean

# ─── Docker ───────────────────────────────────────────────────────────
up:
	docker compose up --build

down:
	docker compose down

# ─── Local dev (no Docker) ────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

# ─── Quality ──────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

lint:
	ruff check . && black --check . && isort --check .

# ─── Reproducibility ─────────────────────────────────────────────────
reproduce:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist frontend/node_modules/.vite 2>/dev/null || true
