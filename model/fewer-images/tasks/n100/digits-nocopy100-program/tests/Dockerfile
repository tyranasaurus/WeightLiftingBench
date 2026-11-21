FROM python:3.12.8-slim-bookworm
COPY requirements.txt /opt/wsp/requirements.txt
RUN pip install --no-cache-dir -r /opt/wsp/requirements.txt
COPY wsp-src /opt/wsp/src
ENV PYTHONPATH=/opt/wsp/src
COPY --chmod=755 test.sh /tests/test.sh
COPY verifier_config.json /tests/verifier_config.json
COPY score_program.py /tests/score_program.py
COPY rebuild.py /tests/rebuild.py
COPY hidden.npz /tests/hidden.npz
WORKDIR /app
