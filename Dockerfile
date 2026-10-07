FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/preprocess.py src/preprocess.py
COPY api api
COPY models/model.joblib models/model.joblib

EXPOSE 8000

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "api.main:app"]