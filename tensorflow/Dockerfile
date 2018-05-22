FROM tensorflow/tensorflow:1.7.1-gpu

RUN mkdir /workspace && cd /workspace && \
apt-get update && apt-get install -y git && \
git clone https://github.com/tensorflow/benchmarks.git
