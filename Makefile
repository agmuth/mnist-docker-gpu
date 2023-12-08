install:
	poetry install

format:
	poetry run black app/
	poetry run black src/

	poetry run isort app/
	poetry run isort src/

lint:
	poetry run ruff --fix app/
	poetry run ruff --fix src/
