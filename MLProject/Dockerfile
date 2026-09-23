FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY mlruns /app/mlruns

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.22.0 \
    cloudpickle==3.1.1 \
    numpy==2.1.3 \
    pandas==2.2.3 \
    psutil==5.9.0 \
    scikit-learn==1.6.1 \
    scipy==1.15.2 \
    seaborn==0.11.2

EXPOSE 5000

CMD ["mlflow", "models", "serve", "--model-uri", "/app/mlruns/0/MODEL_RUN_ID/artifacts/model", "--host", "0.0.0.0", "--port", "5000"]
