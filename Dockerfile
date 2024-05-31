FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install hydra-core
RUN pip install prometheus_client
RUN pip install rich

RUN tar xvfz prometheus-2.45.5.linux-amd64.tar.gz && \
mv prometheus-2.45.5.linux-amd64 prometheus
# Expose ports for Prometheus and the application

# Copy Prometheus configuration file
COPY prometheus.yml /app/prometheus/

EXPOSE 9090 8000

# Start Prometheus
CMD ./prometheus/prometheus --config.file=./prometheus/prometheus.yml & python -m cProfile -s cumtime src/scripts/MLOPSproject.py 
#& python -m cProfile -s tottime -o data/cprofilerOutput.txt src/scripts/MLOPSproject.py


#RUN pip install rich
#CMD python -m cProfile -s tottime -o data/cprofilerOutput.txt src/scripts/MLOPSproject.py

#CMD python -m cProfile -s cumtime src/scripts/MLOPSproject.py

