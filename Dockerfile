FROM python:3.11-slim

WORKDIR /app

COPY vectyfi_src/ vectyfi_src/
COPY requirements-api.txt requirements-api.txt
COPY setup.py setup.py

RUN pip install --upgrade pip && pip install -e .

CMD uvicorn vectyfi_src.api.fast:app --host 0.0.0.0 --port ${PORT:-8000}
