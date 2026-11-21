FROM python:3.12.8-slim-bookworm
RUN pip install --no-cache-dir numpy==2.5.4
COPY --chmod=755 test.sh /tests/test.sh
COPY hidden.npz blocks.py check_network.py task_shape.json score_design.py rebuild.py /tests/
WORKDIR /app
