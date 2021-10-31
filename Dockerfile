ARG PYTHON_ENV=python:3.8.7-slim
FROM $PYTHON_ENV as build

RUN apt-get update
RUN apt-get install -y build-essential swig mecab libmecab-dev mecab-ipadic-utf8

RUN pip install -U pip poetry

RUN mkdir -p /app
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-dev --no-interaction

FROM $PYTHON_ENV as prod

RUN apt-get update && \
    apt-get install -y mecab mecab-ipadic-utf8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH

EXPOSE 8080

CMD ["bash"]
