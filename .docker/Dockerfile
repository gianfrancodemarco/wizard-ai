ARG BUILD_TYPE
FROM tiangolo/uvicorn-gunicorn:python3.11 AS base

WORKDIR /app/

ADD mai_assistant/requirements.txt .
RUN pip install -r requirements.txt

ADD mai_assistant mai_assistant
ADD mai_assistant/main.py .

ENV PYTHONPATH=/app

ARG BUILD_TYPE
RUN echo "Building for ${BUILD_TYPE}"

FROM base AS build_development
ENTRYPOINT ["/start-reload.sh"]

FROM base AS build_testing
FROM base as build_production
ENTRYPOINT ["/start.sh"]

FROM build_${BUILD_TYPE}