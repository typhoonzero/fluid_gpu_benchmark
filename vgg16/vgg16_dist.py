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

import vgg16_model
from .. import env_config

if __name__ == "__main__":
    train, trainp = vgg16_model.get_trainer(parallel=False)
    conf = env_config.config_dict

    eplist = []
    pserver_ips = conf["paddle_init_pservers"]
    port = str(conf["paddle_init_port"])
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)
    trainers = conf["trainers"]
    trainer_id = conf.get("paddle_init_trainer_id", "0")
    current_endpoint = conf["pod_ip"] + ":" + port
    role = conf.get("training_role", "TRAINER")
    print("role: %s endpoints: %s, trainers: %d, trainer_id: %d, current: %s" %\
        (role, pserver_endpoints, trainers, trainer_id, current_endpoint))

    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id,
        pservers=pserver_endpoints,
        trainers=trainers)

    if role == "PSERVER":
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        pserver_prog = t.get_pserver_program(current_endpoint)
        pserver_startup = t.get_startup_program(current_endpoint,
                                                pserver_prog)
        exe.run(pserver_startup)
        exe.run(pserver_prog)
    elif role == "TRAINER":
        trainer_prog = t.get_trainer_program()
        train(trainer_prog)
        # for multi GPU:
        # trainp(trainer_prog, True)
