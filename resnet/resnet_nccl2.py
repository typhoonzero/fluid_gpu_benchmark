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

import paddle
import paddle.fluid as fluid
import contextlib
import math
import sys
import numpy
import unittest
import os
import numpy as np
import time


def resnet_cifar10(input, depth=50):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def train(use_cuda):
    classdim = 102
    data_shape = [3, 224, 224]

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    net = resnet_cifar10(images, 50)

    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)

    # Test program 
    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_cost)

    BATCH_SIZE = 20
    PASS_NUM = 50

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.flowers.train(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=BATCH_SIZE)

    num_trainers = 1
    trainer_id = 0
    # ========================= for nccl2 dist train =================================
    if os.getenv("PADDLE_INIT_TRAINER_ID", None) != None:
        # append gen_nccl_id at the end of startup program
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        print("using trainer_id: ", trainer_id)
        port = os.getenv("PADDLE_INIT_PORT")
        worker_ips = os.getenv("PADDLE_WORKERS")
        worker_endpoints = []
        for ip in worker_ips.split(","):
            print(ip)
            worker_endpoints.append(':'.join([ip, port]))
        num_trainers = len(worker_endpoints)
        current_endpoint = os.getenv("POD_IP") + ":" + port
        worker_endpoints.remove(current_endpoint)

        nccl_id_var = fluid.default_startup_program().global_block().create_var(
            name="NCCLID",
            persistable=True,
            type=fluid.core.VarDesc.VarType.RAW
        )
        fluid.default_startup_program().global_block().append_op(
            type="gen_nccl_id",
            inputs={},
            outputs={"NCCLID": nccl_id_var},
            attrs={"endpoint": current_endpoint,
                   "endpoint_list": worker_endpoints,
                   "trainer_id": trainer_id}
        )
    # ========================= for nccl2 dist train =================================

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

    def test(pass_id, exe):
        acc_list = []
        avg_loss_list = []
        for tid, test_data in enumerate(test_reader()):
            loss_t, acc_t = exe.run(program=test_program,
                                    feed=feeder.feed(test_data),
                                    fetch_list=[avg_cost, acc])
            if math.isnan(float(loss_t)):
                sys.exit("got NaN loss, training failed.")
            acc_list.append(float(acc_t))
            avg_loss_list.append(float(loss_t))

        acc_value = numpy.array(acc_list).mean()
        avg_loss_value = numpy.array(avg_loss_list).mean()

        print(
            'PassID {0:1}, Test Loss {1:2.2}, Acc {2:2.2}'.
            format(pass_id,
                   float(avg_loss_value), float(acc_value)))

    def train_loop(main_program):
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for pass_id in range(PASS_NUM):
            pass_start = time.time()
            num_samples = 0
            for batch_id, data in enumerate(train_reader()):
                exe.run(main_program, feed=feeder.feed(data))
                num_samples += len(data)
            pass_spent = time.time() - pass_start
            print("Pass id %d, train speed %f" % (pass_id, num_samples / pass_spent))
            test(pass_id, exe)
    
    def train_loop_parallel(main_program):
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())
        exe = fluid.ParallelExecutor(True, avg_cost.name, num_threads=1,
                                    allow_op_delay=False,
                                    num_trainers=num_trainers, trainer_id=trainer_id)

        feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
        for pass_id in range(PASS_NUM):
            num_samples = 0
            start_time = time.time()
            for batch_id, data in enumerate(train_reader()):
                loss, = exe.run(
                        [avg_cost.name],
                        feed=feeder.feed(data))
                num_samples += len(data)
                if batch_id % 1 == 0:
                    print("Pass %d, batch %d, loss %s" % (pass_id, batch_id, np.array(loss)))
            pass_elapsed = time.time() - start_time
            print(
                "Pass = %d, Training performance = %f imgs/s\n"
                % (pass_id, num_samples / pass_elapsed))
            test(pass_id, startup_exe)

    train_loop_parallel(fluid.default_main_program())

if __name__ == '__main__':
    train(True)

