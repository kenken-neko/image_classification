# Setup base container
ARG PYTHON_ENV=python:3.8.7-slim
FROM $PYTHON_ENV

# Setup application
COPY pyproject.toml .
COPY poetry.lock .
RUN mkdir -p /root/.config/pypoetry/ && \
    touch /root/.config/pypoetry/config.toml
RUN pip install poetry && \
    poetry config --local virtualenvs.create false && \
    poetry install -v

WORKDIR /app
COPY . .
CMD ["bash"]
