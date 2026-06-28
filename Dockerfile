FROM tensorflow/tensorflow:latest-gpu
# Нужны для read_parquet
RUN pip install --no-cache-dir pandas pyarrow
WORKDIR /work
