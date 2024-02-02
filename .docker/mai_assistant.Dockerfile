ARG BUILD_TYPE
FROM tiangolo/uvicorn-gunicorn:python3.10 AS base

WORKDIR /app/

COPY mai_assistant/pyproject.toml mai_assistant/poetry.lock ./

# Install Poetry and dependencies, don't install the project itself
RUN pip install poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-root

# Poetry install debugpy
RUN poetry add debugpy --dev

COPY mai_assistant .

# Install the project
RUN poetry install --only-root

ENV PYTHONPATH=/app

ARG BUILD_TYPE
RUN echo "Building for ${BUILD_TYPE}"

# Setting debug manually because skaffold debug is not working
# Set env for container
FROM base AS build_development
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "mai_assistant.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
COPY client_secret.json /

FROM base AS build_testing
ENTRYPOINT ["python", "-m", "uvicorn", "mai_assistant.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS build_production
ENTRYPOINT ["python", "-m", "uvicorn", "mai_assistant.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM build_${BUILD_TYPE}