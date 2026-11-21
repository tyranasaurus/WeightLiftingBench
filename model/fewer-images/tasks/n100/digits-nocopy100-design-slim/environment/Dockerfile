FROM python:3.12.8-slim-bookworm
RUN pip install --no-cache-dir numpy==2.5.4
RUN mkdir -p /app/out /app/data
WORKDIR /app
COPY data/train.npz /app/data/train.npz
COPY blocks.py check_network.py run_network.py task_shape.json /app/
COPY rebuild.py /app/check_rebuild.py
