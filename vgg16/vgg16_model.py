# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

"""VGG16 benchmark in Fluid"""
import sys
import time
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
import argparse
import functools
import os
from .. import env_config

def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=4096, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=4096, act=None)
    return fc2


def get_trainer(parallel=False):
    conf = {
        "use_gpu": True,
        "data_set": "cifar10",
        "data_format": "NCHW",
        "batch_size": 128,
        "num_passes": 50
    }
    conf.update(env_config.get_config())
    env_config.config_dict = conf
    
    if conf["data_set"] == "cifar10":
        classdim = 10
        if conf["data_format"] == 'NCHW':
            data_shape = [3, 32, 32]
        else:
            data_shape = [32, 32, 3]
    else:
        classdim = 102
        if conf["data_format"] == 'NCHW':
            data_shape = [3, 224, 224]
        else:
            data_shape = [224, 224, 3]

    # Input data
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    net = vgg16_bn_drop(images)
    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size_tensor)

    # inference program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
                            target_vars=[batch_acc, batch_size_tensor])

    # Optimization
    optimizer = fluid.optimizer.Adam(learning_rate=conf["learning_rate"])
    optimizer.minimize(avg_cost)

    # data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10() if conf["data_set"] == 'cifar10'
            else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=conf["batch_size"])
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if conf["data_set"] == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=conf["batch_size"])

    # test
    def test(exe):
        test_accuracy = fluid.average.WeightedAverage()
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            acc, weight = exe.run(inference_program,
                            feed={"pixel": img_data,
                                  "label": y_data},
                            fetch_list=[batch_acc, batch_size_tensor])
            test_accuracy.add(value=acc, weight=weight)
        return test_accuracy.eval()

    def train_loop(trainer_prog):
        place = core.CPUPlace() if not conf["use_gpu"] else core.CUDAPlace(0)
        iters = 0
        accuracy = fluid.average.WeightedAverage()
        start_time = time.time()
        num_samples = 0
        accuracy.reset()
        feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())

        for pass_id in range(conf["num_passes"]):
            # train
            start_time = time.time()
            num_samples = 0
            with profiler.profiler("All", 'total') as prof:
                for batch_id, data in enumerate(train_reader()):
                    batch_st = time.time()
                    loss, acc, weight = exe.run(
                        trainer_prog,
                        feed=feeder.feed(data),
                        fetch_list=[avg_cost, batch_acc, batch_size_tensor])
                    accuracy.add(value=acc, weight=weight)
                    iters += 1
                    num_samples += len(data)
                    print(
                        "Pass = %d, Iters = %d, Loss = %f, Accuracy = %f, batch spent %f" %
                        (pass_id, iters, loss, acc, time.time() - batch_st)
                    )  # The accuracy is the accumulation of batches, but not the current batch.

            pass_elapsed = time.time() - start_time
            pass_train_acc = accuracy.eval()
            pass_test_acc = test(exe)
            print(
                "Pass = %d, Training performance = %f imgs/s, Train accuracy = %f, Test accuracy = %f\n"
                % (pass_id, num_samples / pass_elapsed, pass_train_acc,
                pass_test_acc))

    def train_loop_parallel(trainer_prog, bcast=False):
        place = core.CPUPlace() if not conf["use_gpu"] else core.CUDAPlace(0)
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())
        exe = fluid.ParallelExecutor(conf["use_gpu"], avg_cost.name)

        feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

        for pass_id in range(conf["num_passes"]):
            num_samples = 0
            start_time = time.time()
            for batch_id, data in enumerate(train_reader()):
                loss, = exe.run(
                        [avg_cost.name],
                        feed=feeder.feed(data))
                num_samples += len(data)
                if bcast:
                    exe.bcast_params()
                print("Pass %d, batch %d, loss %s" % (pass_id, batch_id, np.array(loss)))
            print("Pass avg speed: %f", num_samples / (time.time() - start_time))

    return train_loop, train_loop_parallel
    # if parallel:        
    #     train_loop_parallel(True, fluid.default_main_program())
    # else:
    #     train_loop(True, fluid.default_main_program())
