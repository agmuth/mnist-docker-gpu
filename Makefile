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
	docker image build -t mnist-gradio .

container:
	docker container run -d --name mnist-gradio --gpus all -p 7860:7860 mnist-gradio
