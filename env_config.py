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

import os

config_dict = {
    "batch_size": 20,
    "learning_rate": 1e-3,
    "num_passes": 50
}

ignore_envs = ["ld_library_path", "pythonpath",
               "pwd", "lang", "path", "language", "_",
               "ls_colors", "lc_all", "term",
               "library_path", "hostname", "oldpwd"]

def get_config(do_print=True):
    for k, v in os.environ.iteritems():
        if k.lower() in ignore_envs:
            continue
        typed_value = None
        if v in ["TRUE", "true", "True"]:
            typed_value = True
        elif v in ["FALSE", "false", "False"]:
            typed_value = False
        elif v.isdigit() :
            typed_value = int(v)
        else:
            try:
                typed_value = float(v)
            except:
                typed_value = v
        config_dict[k.lower()] = typed_value
    if do_print:
        print('-----------  Configuration Arguments -----------')
        for k, v in config_dict.iteritems():
            print("%s\t%s" % (k, v))
        print('------------------------------------------------')
    return config_dict