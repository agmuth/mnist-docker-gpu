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

build:
	poetry run python src/build.py $(config)

image:
	docker build -t mnist-gradio .

container:
	docker run -d --name mnist-gradio -p 7860:7860 mnist-gradio
