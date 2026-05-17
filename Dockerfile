FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir "urllib3<2.0.0"

EXPOSE 5000

CMD ["python", "main.py"]