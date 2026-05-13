.PHONY: frontend backend docker-up docker-down healthcheck

frontend:
	cd frontend && npm install && npm run dev

backend:
	cd backend && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up -d

docker-down:
	docker compose down

healthcheck:
	bash scripts/healthcheck.sh
