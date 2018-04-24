#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import argparse
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.v2 as paddle
import sys
import numpy
import unittest
import math
import sys
import os

BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def mlp(img, label):
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    # hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    return loss_net(hidden, label)


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(nn_type,
          use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None,
          is_local=os.getenv("LOCAL") == "TRUE"):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if nn_type == 'mlp':
        net_conf = mlp
    else:
        net_conf = conv_net

    prediction, avg_loss, acc = net_conf(img, label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimize_ops, params_grads = optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        print("###### run startup ok ########")

        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                # train a mini-batch, fetch nothing
                exe.run(main_program, feed=feeder.feed(data))
                if (batch_id + 1) % 10 == 0:
                    acc_set = []
                    avg_loss_set = []
                    for test_data in test_reader():
                        acc_np, avg_loss_np = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=[acc, avg_loss])
                        acc_set.append(float(acc_np))
                        avg_loss_set.append(float(avg_loss_np))
                    # get test acc and loss
                    acc_val = numpy.array(acc_set).mean()
                    avg_loss_val = numpy.array(avg_loss_set).mean()
                    print(
                        'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}'.
                        format(pass_id, batch_id + 1,
                               float(avg_loss_val), float(acc_val)))
                    if math.isnan(float(avg_loss_val)):
                        sys.exit("got NaN loss, training failed.")
        raise AssertionError("Loss of recognize digits is too large")

    def train_loop_parallel(use_gpu, bcast=False):
        place = core.CPUPlace() if not use_gpu else core.CUDAPlace(0)
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())
        print("init parallel exe...")
        exe = fluid.ParallelExecutor(use_gpu, avg_loss.name, num_threads=1)

        for pass_id in range(1000):
            for batch_id, data in enumerate(train_reader()):
                loss, = exe.run(
                        [avg_loss.name],
                        feed=feeder.feed(data))
                if bcast:
                    exe.bcast_params()
                print("Pass %d, batch %d, loss %s" % (pass_id, batch_id, numpy.array(loss)))

    if is_local:
        # train_loop(fluid.default_main_program())
        exe.run(fluid.default_startup_program())
        train_loop_parallel(True)
    else:
        # pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # all pserver endpoints
        # eplist = []
        # port = os.getenv("PADDLE_INIT_PORT")
        # for ip in pserver_ips.split(","):
        #     eplist.append(':'.join([ip, port]))
        # pserver_endpoints = ",".join(eplist)
        pserver_endpoints =  os.getenv("PSERVERS")
        print("pserver endpoints: ", pserver_endpoints)
        trainers = int(os.getenv("TRAINERS"))  # total trainer count
        print("trainers total: ", trainers)
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID", "0"))
        # current_endpoint = os.getenv(
        #     "POD_IP") + ":" + port  # current pserver endpoint
        current_endpoint = os.getenv("CURRENT_ENDPOINT")
        training_role = os.getenv(
            "TRAINING_ROLE",
            "TRAINER")  # get the training role: trainer/pserver
        print("############# program to transpils ############")
        print(fluid.default_main_program())
        print("############# program to transpils ############")
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            server_exe = fluid.Executor(fluid.CPUPlace())
            server_exe.run(pserver_startup)
            server_exe.run(pserver_prog)
        elif training_role == "TRAINER":
            t.get_trainer_program()
            train_loop_parallel(True)

if __name__ == '__main__':
    train("mlp",
          True  # use_cuda
    )

