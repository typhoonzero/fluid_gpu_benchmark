import numpy as np
import sys
import os
import argparse
import time

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

from config import TrainConfig as conf

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dict_path',
        type=str,
        required=True,
        help="Path of the word dictionary.")
    parser.add_argument(
        '--local',
        type=str2bool,
        default=False,
        help="whether to run as local mode.")
    return parser.parse_args()


# Define to_lodtensor function to process the sequential data.
def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


# Load the dictionary.
def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


# Define the convolution model.
def conv_net(dict_dim,
             window_size=3,
             emb_dim=128,
             num_filters=128,
             fc0_dim=96,
             class_dim=2):

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim], is_sparse=False)

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=num_filters,
        filter_size=window_size,
        act="tanh",
        pool_type="max")

    fc_0 = fluid.layers.fc(input=[conv_3], size=fc0_dim)

    prediction = fluid.layers.fc(input=[fc_0], size=class_dim, act="softmax")

    cost = fluid.layers.cross_entropy(input=prediction, label=label)

    avg_cost = fluid.layers.mean(x=cost)

    return data, label, prediction, avg_cost


def main(dict_path):
    word_dict = load_vocab(dict_path)
    word_dict["<unk>"] = len(word_dict)
    dict_dim = len(word_dict)
    print("The dictionary size is : %d" % dict_dim)

    data, label, prediction, avg_cost = conv_net(dict_dim)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=conf.learning_rate)
    optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)

    batch_size_var = fluid.layers.create_tensor(dtype='int64')
    batch_acc_var = fluid.layers.accuracy(
        input=prediction, label=label, total=batch_size_var)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[batch_acc_var, batch_size_var])

    # The training data set.
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=51200),
        batch_size=conf.batch_size)

    # The testing data set.
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.test(word_dict), buf_size=51200),
        batch_size=conf.batch_size)

    if conf.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    # exe.run(fluid.default_startup_program())

    train_pass_acc_evaluator = fluid.average.WeightedAverage()
    test_pass_acc_evaluator = fluid.average.WeightedAverage()

    def test(exe):
        test_pass_acc_evaluator.reset()
        for batch_id, data in enumerate(test_reader()):
            input_seq = to_lodtensor(map(lambda x: x[0], data), place)
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])
            b_acc, b_size = exe.run(inference_program,
                                    feed={"words": input_seq,
                                          "label": y_data},
                                    fetch_list=[batch_acc_var, batch_size_var])
            test_pass_acc_evaluator.add(value=b_acc, weight=b_size)
        test_acc = test_pass_acc_evaluator.eval()
        return test_acc

    def train_loop(exe, train_program, trainer_id):
        total_time = 0.
        for pass_id in xrange(conf.num_passes):
            train_pass_acc_evaluator.reset()
            start_time = time.time()
            total_samples = 0
            with profiler.profiler("CPU", 'total', profile_path='./profile_res_%d' % trainer_id) as prof:
                for batch_id, data in enumerate(train_reader()):
                    batch_start = time.time()
                    cost_val, acc_val, size_val = exe.run(
                        train_program,
                        feed=feeder.feed(data),
                        fetch_list=[avg_cost, batch_acc_var, batch_size_var])
                    train_pass_acc_evaluator.add(value=acc_val, weight=size_val)
                    total_samples += float(size_val)
                    if batch_id and batch_id % conf.log_period == 0:
                        print("Pass id: %d, batch id: %d, cost: %f, pass_acc: %f, speed: %f, time: %f" %
                            (pass_id, batch_id, cost_val,
                            train_pass_acc_evaluator.eval(),
                            float(size_val) / (time.time() - batch_start), time.time() - batch_start))
            end_time = time.time()
            total_time += (end_time - start_time)
            pass_test_acc = test(exe)
            print("Pass id: %d, test_acc: %f, speed: %f" % (pass_id, pass_test_acc, total_samples / (end_time - start_time)))
        print("Total train time: %f" % (total_time))

    if args.local:
        print("run as local mode")
        exe.run(fluid.default_startup_program())
        train_loop(exe, fluid.default_main_program(), 0)
    else:
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # all pserver endpoints
        eplist = []
        port = os.getenv("PADDLE_INIT_PORT")
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        print("pserver endpoints: ", pserver_endpoints)
        trainers = int(os.getenv("TRAINERS"))  # total trainer count
        print("trainers total: ", trainers)
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID", "0"))
        current_endpoint = os.getenv(
            "POD_IP") + ":" + port  # current pserver endpoint
        training_role = os.getenv(
            "TRAINING_ROLE",
            "TRAINER")  # get the training role: trainer/pserver
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)

        if training_role == "PSERVER":
            if not current_endpoint:
                print("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            with open("/tmp/pserver_prog", "w") as f:
                f.write(pserver_prog.__str__())
            print("######## pserver prog in /tmp/pserver_prog #############")
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            print("starting server side startup")
            exe.run(pserver_startup)
            print("starting parameter server...")
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            trainer_prog = t.get_trainer_program()
            with open("/tmp/trainer_prog", "w") as f:
                f.write(trainer_prog.__str__())
            print("######## trainer prog in /tmp/trainer_prog #############")
            # TODO(typhoonzero): change trainer startup program to fetch parameters from pserver
            exe.run(fluid.default_startup_program())
            train_loop(exe, trainer_prog, trainer_id)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")

if __name__ == '__main__':
    args = parse_args()
    main(args.dict_path)
