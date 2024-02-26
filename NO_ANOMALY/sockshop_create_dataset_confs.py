import json
import os
import shutil

import pandas as pd

import data_preparation
import performance_evaluator
import utils
import data_preparation


def __main__(dir_input, dir_output):
    for conf in os.listdir(dir_input):
        exp_dir = os.listdir(os.path.join(dir_input, conf))[0]
        path_src = os.path.join(dir_input, conf, exp_dir)
        path_dst = os.path.join(dir_output, exp_dir)
        if os.path.exists(path_dst):
            u_dir = os.listdir(path_src)[0]
            path_src = os.path.join(path_src, u_dir)
            path_dst = os.path.join(path_dst, u_dir)

            if os.path.exists(path_dst):
                l_dir = os.listdir(path_src)[0]
                path_src = os.path.join(path_src, l_dir)
                path_dst = os.path.join(path_dst, l_dir)

                if not os.path.exists(path_dst):
                    shutil.move(path_src, path_dst)
                else:
                    print("ERRORE")
                    print(conf)
                    # return
            else:
                shutil.move(path_src, path_dst)
        else:
            shutil.move(path_src, path_dst)

    with open("../sockshop/mapping_service_request.json", 'r') as f_map:
        mapping = json.load(f_map)
        with open("../sockshop/pods.txt") as f_pods:
            pods = [p.replace("\n", "") for p in f_pods.readlines()]
            df = data_preparation.read_experiments(dir_output, mapping, pods)

            df.to_csv(os.path.join(dir_output, "df.csv"), index=False)
