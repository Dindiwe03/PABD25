FROM python:3.12-slim

WORKDIR /app


COPY service/requirements.txt .


RUN apt-get update && apt-get install -y gcc && \
    pip install --no-cache-dir -r requirements.txt


COPY service/ .


ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--workers=1", "--bind=0.0.0.0:5000", "app:app"]