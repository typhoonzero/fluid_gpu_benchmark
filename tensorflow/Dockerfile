FROM tensorflow/tensorflow:1.7.1-gpu

#ENV http_proxy=http://172.19.32.166:9988/
#ENV https_proxy=http://172.19.32.166:9988/

RUN apt-get update && apt-get install -y git

RUN mkdir /workspace && cd /workspace && \
git clone https://github.com/tensorflow/benchmarks.git

RUN pip install kubernetes
ADD libcudnn.so.7.0.5 /usr/lib/x86_64-linux-gnu/
RUN rm -f /usr/lib/x86_64-linux-gnu/libcudnn.so.7 && ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7.0.5 /usr/lib/x86_64-linux-gnu/libcudnn.so.7
ADD paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root
ADD run_dist_train.sh /root
