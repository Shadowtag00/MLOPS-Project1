FROM python:3-alpine3.20
COPY . /app
WORKDIR /app
CMD python MLOPSproject.py