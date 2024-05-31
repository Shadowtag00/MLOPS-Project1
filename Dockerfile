FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install rich
RUN pip install mlflow
RUN mlflow server --host localhost --port 9000 &
CMD python -m cProfile -s tottime -o data/cprofilerOutput.txt src/scripts/MLOPSproject.py
#CMD python -m cProfile -s tottime src/scripts/MLOPSproject.py