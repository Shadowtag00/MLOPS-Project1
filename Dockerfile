FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install pandas
RUN pip install scikit-learn
CMD python src/scripts/MLOPSproject.py