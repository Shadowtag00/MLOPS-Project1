FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install rich
RUN pip install torch torchvision
CMD python src/scripts/MLOPSproject.py