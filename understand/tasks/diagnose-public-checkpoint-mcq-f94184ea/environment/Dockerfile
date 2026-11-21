FROM python:3.12.8-slim-bookworm
RUN pip install --no-cache-dir numpy==2.5.4
RUN mkdir -p /app/out
WORKDIR /app
COPY policy.npz /app/policy.npz
