FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV https_proxy=http://172.19.32.166:8899/
ENV http_proxy=http://172.19.32.166:8899/

RUN apt-get update && apt-get install -y python python-pip iputils-ping libgtk2.0-dev wget
RUN pip install -U pip
RUN pip install -U kubernetes opencv-python
# NOTE: By default CI built wheel packages turn WITH_DISTRIBUTE=OFF,
#       so we must build one with distribute support to install in this image.

RUN pip install paddlepaddle
#ADD common.py /usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/common.py
RUN cat /usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/flowers.py
RUN sed -i "s/52808999861908f626f3c1f4e79d11fa/33bfc11892f1e405ca193ae9a9f2a118/g" /usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/flowers.py && cat /usr/local/lib/python2.7/dist-packages/paddle/v2/dataset/flowers.py


RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()\npaddle.dataset.flowers.train()" | python'
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.mnist.train()\npaddle.dataset.mnist.test()\npaddle.dataset.imdb.fetch()" | python'
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.imikolov.fetch()" | python'
RUN pip uninstall -y paddlepaddle
RUN apt-get install -y vim net-tools iftop
RUN mkdir /workspace && wget -q -O /workspace/aclImdb_v1.tar.gz http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz && cd /workspace && tar zxf aclImdb_v1.tar.gz && cd -

# below lines may change a lot for debugging
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root
RUN apt-get install -y libnccl2=2.1.2-1+cuda8.0
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl && \
chmod +x /usr/bin/paddle_k8s
ENV http_proxy=""
ENV https_proxy=""

RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.5 /usr/lib/libcudnn.so && ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/libnccl.so

ADD paddle_k8s /usr/bin
# patch
ENV LD_LIBRARY_PATH=/usr/local/lib
ADD seq_tag_ner /workspace/seq_tag_ner
ADD text_classification /workspace/text_classification
ADD lm /workspace/lm
ADD vgg16 /workspace/vgg16
