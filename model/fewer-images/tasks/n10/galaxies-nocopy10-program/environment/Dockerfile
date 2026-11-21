FROM python:3.12.8-slim-bookworm
RUN pip install --no-cache-dir numpy==2.5.4
RUN mkdir -p /app/out /app/l0 /app/data
WORKDIR /app
COPY data/train.npz /app/data/train.npz
COPY answer.json /app/l0/answer.json
COPY check_program.py /app/check_program.py
COPY run_program.py /app/run_program.py
COPY rebuild.py /app/check_rebuild.py
COPY task_limits.json /app/task_limits.json
