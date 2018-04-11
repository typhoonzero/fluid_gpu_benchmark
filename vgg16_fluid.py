#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import print_function

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size', type=int, default=128, help="Batch size for training.")
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help="Learning rate for training.")
parser.add_argument('--num_passes', type=int, default=50, help="No. of passes.")
parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    '--data_format',
    type=str,
    default='NCHW',
    choices=['NCHW', 'NHWC'],
    help='The data order, now only support NCHW.')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    choices=['cifar10', 'flowers'],
    help='Optional dataset for benchmark.')
parser.add_argument(
    '--local',
    type=str2bool,
    default=True,
    help='Whether to run as local mode.')
args = parser.parse_args()


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
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


def main():
    if args.data_set == "cifar10":
        classdim = 10
        if args.data_format == 'NCHW':
            data_shape = [3, 32, 32]
        else:
            data_shape = [32, 32, 3]
    else:
        classdim = 102
        if args.data_format == 'NCHW':
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
    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimize_ops, params_grads = optimizer.minimize(avg_cost)
    params = [p for p, g in params_grads]

    # data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10() if args.data_set == 'cifar10'
            else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)

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

    def train_loop(use_gpu, trainer_prog, trainer_id=0):
        place = core.CPUPlace() if not use_gpu else core.CUDAPlace(0)
        iters = 0
        accuracy = fluid.average.WeightedAverage()
        start_time = time.time()
        num_samples = 0
        accuracy.reset()
        feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
        exe = fluid.Executor(place)

        for pass_id in range(args.num_passes):
            # train
            start_time = time.time()
            num_samples = 0
            with profiler.profiler("All", 'total', profile_path="/usr/local/nvidia/lib64/tmp") as prof:
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

    def train_loop_parallel(use_gpu, trainer_prog, trainer_id=0, bcast=False):
        place = core.CPUPlace() if not use_gpu else core.CUDAPlace(0)
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())
        exe = fluid.ParallelExecutor(use_gpu, avg_cost.name)

        feeder = fluid.DataFeeder(place=place, feed_list=[images, label])

        for pass_id in range(args.num_passes):
            for batch_id, data in enumerate(train_reader()):
                print("before run one...")
                loss, = exe.run(
                        [avg_cost.name],
                        feed_dict=feeder.feed(data))
                if bcast:
                    exe.bcast_params()
                print("Pass %d, batch %d, loss %s" % (pass_id, batch_id, np.array(loss)))

    def transpile2dist():
        # pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # all pserver endpoints
        # eplist = []
        # port = os.getenv("PADDLE_INIT_PORT")
        # for ip in pserver_ips.split(","):
        #     eplist.append(':'.join([ip, port]))
        # pserver_endpoints = ",".join(eplist)
        pserver_endpoints = os.getenv("PSERVERS")
        print("pserver endpoints: ", pserver_endpoints)
        trainers = int(os.getenv("TRAINERS"))  # total trainer count
        print("trainers total: ", trainers)
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID", "0"))
        # current_endpoint = os.getenv(
        #         "POD_IP") + ":" + port  # current pserver endpoint
        current_endpoint = os.getenv("SERVER_ENDPOINT")
        role = os.getenv(
            "TRAINING_ROLE",
            "TRAINER")  # get the training role: trainer/pserver
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        return t, role, current_endpoint, trainer_id

    use_gpu = False if args.device == 'CPU' else True
    if args.local:
        train_loop_parallel(use_gpu, fluid.default_main_program())
    else:
        t, role, current_endpoint, trainer_id  = transpile2dist()

        if role == "PSERVER":
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            pserver_prog = t.get_pserver_program(current_endpoint)
            # For debug program
            with open("/tmp/pserver_prog", "w") as f:
                f.write(pserver_prog.__str__())
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif role == "TRAINER":
            trainer_prog = t.get_trainer_program()
            # For debug program
            with open("/tmp/trainer_prog", "w") as f:
                f.write(trainer_prog.__str__())
            train_loop_parallel(use_gpu, trainer_prog, trainer_id, True)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    print_arguments()
    main()
