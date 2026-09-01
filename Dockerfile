FROM python:3.12-slim

WORKDIR /app

RUN apt-get update

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pytest

COPY . .

ENV PYTHONPATH=/app

CMD ["pytest", "einn/tests/"]
