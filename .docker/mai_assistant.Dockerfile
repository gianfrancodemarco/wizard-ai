ARG BUILD_TYPE
FROM tiangolo/uvicorn-gunicorn:python3.10 AS base

WORKDIR /app/

ADD mai_assistant/requirements.txt .
RUN pip install -r requirements.txt

ADD mai_assistant mai_assistant
ADD mai_assistant/main.py .
ADD client_secret.json /

ENV PYTHONPATH=/app

ARG BUILD_TYPE
RUN echo "Building for ${BUILD_TYPE}"

FROM base AS build_development
RUN pip install debugpy
# Doing it manually because skaffold debug is not working
ENTRYPOINT [ "python", "-u", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn",  "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS build_testing
FROM base as build_production
ENTRYPOINT [ "python", "-m", "uvicorn",  "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM build_${BUILD_TYPE}