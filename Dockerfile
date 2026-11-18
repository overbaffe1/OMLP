# Образ для сервиса и инференса решения «КосмоФАЙР-2».
#   docker build -t cosmofire2 .
#   docker run --rm -v $PWD/data:/data cosmofire2 \
#       python inference.py --data-dir /data/test --output /data/submission.csv --validate
#   docker run --rm -p 8000:8000 -v $PWD/data:/data cosmofire2 \
#       uvicorn service.app:app --host 0.0.0.0 --port 8000
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MPLBACKEND=Agg \
    CATALOG_PATH=/app/service/catalog.json

WORKDIR /app

# сначала зависимости — слой кэшируется при правках кода
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# непривилегированный пользователь: сервис не должен работать от root
RUN useradd --create-home --shell /bin/bash cosmofire && chown -R cosmofire /app
USER cosmofire

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
