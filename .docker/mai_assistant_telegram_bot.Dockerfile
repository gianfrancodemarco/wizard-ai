ARG BUILD_TYPE
FROM python:3.10.13-slim-bookworm AS base

WORKDIR /app/

ADD mai_assistant_telegram_bot/requirements.txt .
RUN pip install -r requirements.txt

ADD mai_assistant_telegram_bot mai_assistant_telegram_bot
ADD mai_assistant_telegram_bot/main.py .

ENV PYTHONPATH=/app

ARG BUILD_TYPE
RUN echo "Building for ${BUILD_TYPE}"

FROM base AS build_development
RUN pip install debugpy
# Doing it manually because skaffold debug is not working
#ENTRYPOINT [ "tail", "-f", "/dev/null"]
ENTRYPOINT [ "python", "-u", "-m", "debugpy", "--listen", "0.0.0.0:5678", "main.py"]

# FROM base AS build_testing
# FROM base as build_production
# ENTRYPOINT [ "python", "main.py"]

FROM build_${BUILD_TYPE}