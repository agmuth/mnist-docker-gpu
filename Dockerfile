# FROM python:3.10-slim

FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

RUN pip install poetry==1.7.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /code

COPY pyproject.toml poetry.lock ./
RUN touch README.md
RUN poetry install --no-root --without dev 

COPY src/ ./src/
RUN poetry install --only-root
RUN rm -rf $POETRY_CACHE_DIR

COPY app/ ./app/

EXPOSE 7860

CMD ["poetry", "run", "python", "app/ui.py"]