import math
import os
import pickle
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework

# set hyper parameters
N = 1000

epoch_num = 20
batch_size = 20

emb_size = 200
hid_size = 200

dropout_rate = 0.5

cuda_id = 0
base_lr = 1.0
emb_lr_x = 10.0

model_dir = "output/model/gru_emb10x"
os.makedirs(model_dir)

cluster_train_file = "./train"
cluster_test_file = "./test"

class DataType(object):
    """
    Definition of datatype
    """
    NGRAM = 1
    SEQ = 2


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


def to_lodtensor(data, place):
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
    
def train_main(use_cuda, is_sparse):
    
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    if training_role == "PSERVER":
        place = fluid.CPUPlace()
    
    # Load dict
    dict_pkl_file = open('./thirdparty/worddict.pkl', 'rb')
    word_dict = pickle.load(dict_pkl_file)
    dict_size = len(word_dict)
    print "dict_size: %d" % dict_size
    
    # build computational graph
    src_wordseq = fluid.layers.data(name="src_wordseq", shape=[1], dtype="int64", lod_level=1)
    dst_wordseq = fluid.layers.data(name="dst_wordseq", shape=[1], dtype="int64", lod_level=1)
    
    emb = fluid.layers.embedding(input=src_wordseq,
            size=[dict_size, emb_size],
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-0.05, high=0.05), learning_rate=emb_lr_x),
            is_sparse=is_sparse)
    
    fc0 = fluid.layers.fc(input=emb, size=hid_size*3, param_attr=fluid.initializer.Uniform(low=-0.05, high=0.05))
    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_size, param_attr=fluid.initializer.Uniform(low=-0.05, high=0.05))
    
    gru_h = fluid.layers.dropout(x=gru_h, dropout_prob=dropout_rate, is_test=False)
    
    fc1 = fluid.layers.fc(input=gru_h, size=dict_size, act='softmax', param_attr=fluid.initializer.Uniform(low=-0.05, high=0.05))
    
    cost = fluid.layers.cross_entropy(input=fc1, label=dst_wordseq)
    average_cost = fluid.layers.mean(x=cost)
    
    sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=base_lr,
                decay_steps=1300*4,
                decay_rate=0.5,
                staircase=True))
    optimize_ops, params_grads = sgd_optimizer.minimize(average_cost)
    
    # run train graph
    exe = fluid.Executor(place)
    
    def train_loop(main_program):
        train_reader = paddle.batch(
                reader_creator(cluster_train_file, word_dict, N, DataType.SEQ), 
                batch_size)
    
        exe.run(fluid.default_startup_program())
        
        for epoch_id in xrange(epoch_num):
            print "epoch_%d start" % epoch_id
        
            i = 0
            for data in train_reader():
                i += 1
                lod_src_wordseq = to_lodtensor(map(lambda x: x[0], data), place)
                lod_dst_wordseq = to_lodtensor(map(lambda x: x[1], data), place)
                ret_average_cost = exe.run(
                        main_program, 
                        feed={"src_wordseq": lod_src_wordseq, "dst_wordseq": lod_dst_wordseq}, 
                        fetch_list=[average_cost])
                average_ppl = math.exp(ret_average_cost[0])
                if i % 100 == 0:
                    print "step %d ppl: %.3f" % (i, average_ppl)
            print "total steps:", i
        
            # save model
            save_dir = "%s/epoch_%d" % (model_dir, epoch_id)
            feed_var_names = ["src_wordseq", "dst_wordseq"]
            fetch_vars = [fc1, average_cost]
            fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
        
            print "epoch_%d finished ." % epoch_id
            
            
    port = os.getenv("PADDLE_PORT", "6174")
    pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
    trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
    current_endpoint = os.getenv("POD_IP") + ":" + port
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
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
        exe.run(pserver_startup)
        exe.run(pserver_prog)
    elif training_role == "TRAINER":
        train_loop(t.get_trainer_program())


if __name__ == '__main__':
    train_main(use_cuda=True, is_sparse=True)