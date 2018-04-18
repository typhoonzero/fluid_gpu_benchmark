import os
import math
import time
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import reader
from network_conf import ner_net
from utils import logger, load_dict
from utils_extend import to_lodtensor, get_embedding


def test(exe, chunk_evaluator, inference_program, test_data, place):
    chunk_evaluator.reset(exe)
    for data in test_data():
        word = to_lodtensor(map(lambda x: x[0], data), place)
        mark = to_lodtensor(map(lambda x: x[1], data), place)
        target = to_lodtensor(map(lambda x: x[2], data), place)
        acc = exe.run(inference_program,
                      feed={"word": word,
                            "mark": mark,
                            "target": target})
    return chunk_evaluator.eval(exe)


def main(train_data_file, test_data_file, vocab_file, target_file, emb_file,
         model_save_dir, num_passes, use_gpu, parallel):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
    word_dict = load_dict(vocab_file)
    label_dict = load_dict(target_file)

    word_vector_values = get_embedding(emb_file)

    word_dict_len = len(word_dict)
    label_dict_len = len(label_dict)

    avg_cost, feature_out, word, mark, target = ner_net(
        word_dict_len, label_dict_len, parallel)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-3 / BATCH_SIZE)
    optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    chunk_evaluator = fluid.evaluator.ChunkEvaluator(
        input=crf_decode,
        label=target,
        chunk_scheme="IOB",
        num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = chunk_evaluator.metrics + chunk_evaluator.states
        inference_program = fluid.io.get_inference_program(test_target)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(train_data_file, word_dict, label_dict),
            buf_size=20000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.data_reader(test_data_file, word_dict, label_dict),
            buf_size=20000),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, mark, target], place=place)
    exe = fluid.Executor(place)


    def train_loop(exe, trainer_prog, trainer_id = 0, reader = train_reader):
        embedding_name = 'emb'
        embedding_param = fluid.global_scope().find_var(embedding_name).get_tensor()
        embedding_param.set(word_vector_values, place)

        batch_id = 0
        for pass_id in xrange(num_passes):
            chunk_evaluator.reset(exe)
            start_time = time.time()
            with profiler.profiler("CPU", 'total', profile_path="/usr/local/nvidia/lib64/tmp") as prof:
                for data in reader():
                    cost, batch_precision, batch_recall, batch_f1_score = exe.run(
                        trainer_prog,
                        feed=feeder.feed(data),
                        fetch_list=[avg_cost] + chunk_evaluator.metrics)
                    if batch_id % 5 == 0:
                        print("Pass " + str(pass_id) + ", Batch " + str(
                            batch_id) + ", Cost " + str(cost[0]) + ", Precision " + str(
                                batch_precision[0]) + ", Recall " + str(batch_recall[0])
                            + ", F1_score" + str(batch_f1_score[0]))
                    batch_id = batch_id + 1

                pass_precision, pass_recall, pass_f1_score = chunk_evaluator.eval(exe)
                spent = time.time() - start_time
                print("pass_id: %d, precision: %f, recall: %f, f1: %f, spent: %f, speed: %f" % \
                      (pass_id, pass_precision, pass_recall, pass_f1_score,
                      spent, 14987.0 / spent))
                pass_precision, pass_recall, pass_f1_score = test(
                    exe, chunk_evaluator, inference_program, test_reader, place)
                print("[TestSet] pass_id:" + str(pass_id) + " pass_precision:" + str(
                    pass_precision) + " pass_recall:" + str(pass_recall) +
                    " pass_f1_score:" + str(pass_f1_score))

                # save_dirname = os.path.join(model_save_dir,
                #     "params_pass_%d_trainer%d" % (pass_id, trainer_id))
                # fluid.io.save_inference_model(save_dirname, ['word', 'mark', 'target'],
                #                             [crf_decode], exe)

    with open("/tmp/origin_prog", "w") as fn:
        fn.write(fluid.default_main_program().__str__())

    if os.getenv("LOCAL") == "TRUE":
        exe.run(fluid.default_startup_program())
        train_loop(exe, fluid.default_main_program())
    else:
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # all pserver endpoints
        eplist = []
        port = os.getenv("PADDLE_INIT_PORT")
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        trainers = int(os.getenv("TRAINERS"))  # total trainer count
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

        print("endpoints: %s, current: %s, trainers: %d, trainer_id: %d, role: %s" %\
              (pserver_endpoints, current_endpoint, trainers, trainer_id, training_role))
        if training_role == "PSERVER":
            if not current_endpoint:
                print("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            print("######## pserver prog #############")
            with open("/tmp/pserver_prog", "w") as f:
                f.write(pserver_prog.__str__())
            print("######## pserver prog #############")
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            with open("/tmp/pserver_startup", "w") as f:
                f.write(pserver_startup.__str__())
            print("starting server side startup")
            exe.run(pserver_startup)
            print("starting parameter server...")
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            exe.run(fluid.default_startup_program())
            trainer_prog = t.get_trainer_program()
            cluster_train_reader = paddle.batch(
                paddle.reader.shuffle(
                    reader.cluster_data_reader(
                        train_data_file, word_dict, label_dict, trainers, trainer_id),
                    buf_size=20000),
                batch_size=BATCH_SIZE)
            print("######## trainer prog #############")
            with open("/tmp/trainer_prog", "w") as f:
                f.write(trainer_prog.__str__())
            print("######## trainer prog #############")
            train_loop(exe, trainer_prog, trainer_id, cluster_train_reader)
        else:
            print("environment var TRAINER_ROLE should be TRAINER os PSERVER")
    


if __name__ == "__main__":
    main(
        train_data_file="data/train",
        test_data_file="data/test",
        vocab_file="data/vocab.txt",
        target_file="data/target.txt",
        emb_file="data/wordVectors.txt",
        model_save_dir="models",
        num_passes=1000,
        use_gpu=False,
        parallel=False)
