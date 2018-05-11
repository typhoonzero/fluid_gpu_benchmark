""" GRU-RNN """

import os
import sys
import math
import time
import pickle
import numpy as np
import paddle.v2 as paddle
import paddle.fluid as fluid

# set hyper parameters
N = 1000

epoch_num = 20
batch_size = int(os.getenv("BATCH_SIZE", "20"))
emb_size = 200
hid_size = 200
cuda_id =0
base_lr = 1.0
emb_lr_x = 10.0
l1_lr_x = 1.0
l2_lr_x = 1.0
fc_lr_x = 1.0
init_bound = 0.1

class DataType(object):
    """
    Definition of datatype
    """
    NGRAM = 1
    SEQ = 2

cluster_train_file = "./train"
cluster_test_file = "./test"

dict_pkl_file = open('./thirdparty/worddict.pkl', 'rb')
word_dict = pickle.load(dict_pkl_file)
dict_size = len(word_dict)
vocab_size = dict_size
print "dict_size: %d" % dict_size


# build computational graph
src_wordseq = fluid.layers.data(name="src_wordseq", shape=[1], dtype="int64", lod_level=1)
dst_wordseq = fluid.layers.data(name="dst_wordseq", shape=[1], dtype="int64", lod_level=1)
emb = fluid.layers.embedding(input=src_wordseq,
        size=[vocab_size, emb_size],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
            learning_rate=emb_lr_x),
        is_sparse=os.getenv("IS_SPARSE", "TRUE") == "TRUE")
fc0 = fluid.layers.fc(input=emb, size=hid_size * 3,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
            learning_rate=l1_lr_x))
gru_h0 = fluid.layers.dynamic_gru(input=fc0, size=hid_size,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
            learning_rate=l1_lr_x))
fc = fluid.layers.fc(input=gru_h0, size=vocab_size, act='softmax',
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
            learning_rate=fc_lr_x))
cost = fluid.layers.cross_entropy(input=fc, label=dst_wordseq)
average_cost = fluid.layers.mean(x=cost)
infer_program = fluid.default_main_program().clone()

# sgd_optimizer = fluid.optimizer.SGD(
#         learning_rate=fluid.layers.exponential_decay(
#             learning_rate=base_lr,
#             decay_steps=2100 * 4,
#             decay_rate=0.5,
#             staircase=True))
momentum_optimizer = fluid.optimizer.Momentum(learning_rate=1.0, momentum=0.1)
optimize_ops, params_grads = momentum_optimizer.minimize(average_cost)


def reader_creator(file_dir, word_idx, n, data_type):
    """
    Create data reader from file
    """
    def reader():
        """
        reader function
        """
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                UNK = word_idx['<unk>']
                for line in f:
                    if DataType.NGRAM == data_type:
                        assert n > -1, 'Invalid gram length'
                        line = ['<s>'] + line.strip().split() + ['<e>']
                        if len(line) >= n:
                            line = [word_idx.get(w, UNK) for w in line]
                            for i in range(n, len(line) + 1):
                                yield tuple(line[i - n:i])
                    elif DataType.SEQ == data_type:
                        line = line.strip().split()
                        line = [word_idx.get(w, UNK) for w in line]
                        src_seq = [word_idx['<s>']] + line
                        trg_seq = line + [word_idx['<e>']]
                        if n > 0 and len(src_seq) > n: 
                            continue
                        yield src_seq, trg_seq
                    else:
                        assert False, 'Unknow data type'

    return reader


def test(exe, pass_id, place):
    cloud_test_reader = paddle.batch(
        reader_creator(cluster_test_file, word_dict, N, DataType.SEQ), 
        batch_size)

    total_loss = 0
    idx = 0
    for data in cloud_test_reader():
        lod_src_wordseq = to_lodtensor(map(lambda x: x[0], data), place)
        lod_dst_wordseq = to_lodtensor(map(lambda x: x[1], data), place)

        loss = exe.run(infer_program,
                    feed={"src_wordseq": lod_src_wordseq, "dst_wordseq": lod_dst_wordseq},
                    fetch_list=[average_cost])
        total_loss += loss[0]
        idx += 1

    ppl = math.exp((total_loss[0] / idx))
    print("Pass %d, test samples: %d, test avg ppl: %.3f" % (pass_id, idx, ppl))

# run train graph
def to_lodtensor(data, place):
    """ help func """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.core.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def main():
    cloud_train_reader = paddle.batch(
        reader_creator(cluster_train_file, word_dict, N, DataType.SEQ), 
        batch_size)
    
    def train_loop(exe, trainer_prog, reader):
        exe.run(fluid.default_startup_program())

        for epoch_id in xrange(epoch_num):
            print "epoch_%d start" % epoch_id
            pass_start = time.time()
            i = 0
            for data in reader():
                i += 1
                lod_src_wordseq = to_lodtensor(map(lambda x: x[0], data), place)
                lod_dst_wordseq = to_lodtensor(map(lambda x: x[1], data), place)
                ret_average_cost = exe.run(
                        trainer_prog,
                        feed={"src_wordseq": lod_src_wordseq, "dst_wordseq": lod_dst_wordseq},
                        fetch_list=[average_cost])
                average_ppl = math.exp(ret_average_cost[0])
                if i % 100 == 0:
                    print "step %d ppl: %.3f" % (i, average_ppl)
            print "total steps:", i
            test(exe, epoch_id, place)
            # NO saving model
            # save_dir = "%s/epoch_%d" % (model_dir, epoch_id)
            # feed_var_names = ["src_wordseq", "dst_wordseq"]
            # fetch_vars = [fc, average_cost]
            # fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
            print "epoch_%d finished, spent: %f" % (epoch_id, time.time() - pass_start)

    def train_loop_parallel(reader):
        place = fluid.CUDAPlace(0)
        feed_place = fluid.CPUPlace()
        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())
        exe = fluid.ParallelExecutor(True, average_cost.name)
        print("#### para_exe running on %d GPUs." % exe.device_count)

        for pass_id in xrange(epoch_num):
            pass_start = time.time()
            for batch_id, data in enumerate(reader()):
                lod_src_wordseq = to_lodtensor(map(lambda x: x[0], data), feed_place)
                lod_dst_wordseq = to_lodtensor(map(lambda x: x[1], data), feed_place)
                loss, = exe.run([average_cost.name],
                        feed={"src_wordseq": lod_src_wordseq, "dst_wordseq": lod_dst_wordseq})
                # broadcast params to all GPU per batch
                exe.bcast_params()
                if batch_id % 100 == 0:
                    print("Pass %d, batch %d, loss %s" % (pass_id, batch_id, np.array(loss)))
            spent = time.time() - pass_start
            print("Pass %d end, spent: %f, speed: %f" % (pass_id, spent, 42068 / spent))
            test(startup_exe, pass_id, place)

    use_gpu = True
    if True:
        pserver_ips = os.getenv("PADDLE_PSERVERS")
        eplist = []
        port = os.getenv("PADDLE_PORT")
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)

        trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))  # total trainer count
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        current_endpoint = os.getenv(
                "POD_IP") + ":" + port  # current pserver endpoint
        role = os.getenv(
            "TRAINING_ROLE",
            "TRAINER")  # get the training role: trainer/pserver

        print("role: %s endpoints: %s, trainers: %d, trainer_id: %d, current: %s" %\
            (role, pserver_endpoints, trainers, trainer_id, current_endpoint))
        
        with open("/tmp/origin_prog", "w") as f:
            f.write(fluid.default_main_program().__str__())

        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)
        if role == "PSERVER":
            # pserver always on CPU
            place = fluid.CPUPlace()
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
            with open("/tmp/trainer_prog", "w") as f:
                f.write(trainer_prog.__str__())
            train_loop_parallel(cloud_train_reader)
        else:
            raise("role %s not supported" % role)

if __name__ == "__main__":
    main()