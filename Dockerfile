FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install hydra-core
RUN pip install prometheus_client
RUN pip install rich
RUN pip install mlflow

#RUN tar xvfz prometheus-2.45.5.linux-amd64.tar.gz && \
#mv prometheus-2.45.5.linux-amd64 prometheus
# Expose ports for Prometheus and the application

# Set the Prometheus version and checksum
ENV PROMETHEUS_VERSION=2.45.5
ENV PROMETHEUS_CHECKSUM="65a61cec978eb44a2a220803a4653e6f1f2dbe69510131a867492981ef39f253"
#RUN echo "testing original"
#RUN tar -xvf "prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz"

# Copy local Prometheus tarball if it exists
#COPY prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz /tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz 

# Check if the local file exists and is valid, else download it
RUN if [ -f "/tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" ]; then \
      echo "Using local Prometheus tarball"; \
    else \
      echo "Downloading Prometheus tarball"; \
      curl -LO https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz && \
      mv "prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" /tmp/; \
    fi && \
    #echo "$PROMETHEUS_CHECKSUM /tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" | sha256sum -c - && \
    tar -xvf "/tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" && \
    echo "extracting file" && \
    #gunzip -c "/tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" | tar -xf - -C /tmp/ && \
    mv "prometheus-$PROMETHEUS_VERSION.linux-amd64" prometheus && \
    rm "/tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz"

# Copy Prometheus configuration file
COPY prometheus.yml /app/prometheus/

EXPOSE 9090 8000 5000

# Start Prometheus
CMD mlflow ui --host 0.0.0.0 --port 5000 & ./prometheus/prometheus --config.file=./prometheus/prometheus.yml & python -m cProfile -s cumtime src/scripts/MLOPSproject.py 
#& python -m cProfile -s tottime -o data/cprofilerOutput.txt src/scripts/MLOPSproject.py


#RUN pip install rich
#CMD python -m cProfile -s tottime -o data/cprofilerOutput.txt src/scripts/MLOPSproject.py

#CMD python -m cProfile -s cumtime src/scripts/MLOPSproject.py

