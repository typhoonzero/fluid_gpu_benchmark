#FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04


RUN apt-get update && apt-get install -y python python-pip iputils-ping libgtk2.0-dev wget vim net-tools iftop
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.5 /usr/lib/libcudnn.so && ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/libnccl.so
RUN pip install -U pip
RUN pip install -U kubernetes opencv-python paddlepaddle

ENV https_proxy=http://172.19.32.166:9988/
ENV http_proxy=http://172.19.32.166:9988/


RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.cifar.train10()\npaddle.dataset.flowers.fetch()" | python'
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.mnist.train()\npaddle.dataset.mnist.test()\npaddle.dataset.imdb.fetch()" | python'
RUN sh -c 'echo "import paddle.v2 as paddle\npaddle.dataset.imikolov.fetch()" | python'
RUN pip uninstall -y paddlepaddle
RUN mkdir /workspace && wget -q -O /workspace/aclImdb_v1.tar.gz http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz && cd /workspace && tar zxf aclImdb_v1.tar.gz && cd -

ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/paddle_k8s /usr/bin
ADD https://raw.githubusercontent.com/PaddlePaddle/cloud/develop/docker/k8s_tools.py /root

ENV https_proxy=
ENV http_proxy=

# below lines may change a lot for debugging
ADD paddle_k8s /usr/bin/paddle_k8s
ADD *.whl /
RUN pip install /*.whl && rm -f /*.whl && chmod +x /usr/bin/paddle_k8s


ENV LD_LIBRARY_PATH=/usr/local/lib
ADD seq_tag_ner /workspace/seq_tag_ner
ADD text_classification /workspace/text_classification
ADD lm /workspace/lm
ADD vgg16 /workspace/vgg16
ADD resnet /workspace/resnet
ADD word2vec /workspace/word2vec
