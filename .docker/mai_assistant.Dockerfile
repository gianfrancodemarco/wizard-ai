ARG BUILD_TYPE
FROM tiangolo/uvicorn-gunicorn:python3.10 AS base

WORKDIR /app/

COPY mai_assistant/pyproject.toml mai_assistant/poetry.lock ./

# Install Poetry and dependencies, don't install the project itself
RUN pip install poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-root

COPY mai_assistant .
COPY client_secret.json .

# Install the project
RUN poetry install --only-root

ENV PYTHONPATH=/app

ARG BUILD_TYPE
RUN echo "Building for ${BUILD_TYPE}"

FROM base AS build_development
# Poetry install debugpy
RUN poetry add debugpy --dev

# Doing it manually because skaffold debug is not working

ENTRYPOINT ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "mai_assistant.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS build_testing
ENTRYPOINT ["python", "-m", "uvicorn", "mai_assistant.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS build_production
ENTRYPOINT ["python", "-m", "uvicorn", "mai_assistant.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM build_${BUILD_TYPE}