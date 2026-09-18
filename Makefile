WORKERS ?= 1

.PHONY: docker-up docker-down docker-status docker-logs

docker-up:
	docker compose up --build -d --scale worker=$(WORKERS)
	docker compose rm -f migrate

docker-down:
	docker compose down

docker-status:
	docker compose ps

docker-logs:
	docker compose logs -f api worker
