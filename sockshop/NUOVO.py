import json
import os
import shutil
import random

import example

path_out = "compact_configs_run"

os.mkdir(path_out)

path = ["works/work_rep",
        "works_mlp/mlp_predictor_rep",
        "works_random/random_predictor_rep"]

for path1 in path:
    for rep in range(20):
        pathrep = path1 + str(rep) + "/run_configs"
        if not os.path.exists(pathrep):
            continue

        lists = os.listdir(pathrep)
        for dir in lists:
            exp_dir = os.listdir(pathrep + "/" + dir)[0]

            shutil.copytree(pathrep + "/" + dir + "/" + exp_dir, path_out + "/" + exp_dir, dirs_exist_ok=True)


example.create_dataset(example.System.SOCKSHOP, path_out + "/df.csv", path_out)