import json
import os
import pandas as pd

start_num = 0


def check(c):
    ddd = pd.read_csv("sockshop/compact_configs_run/df.csv")

    if len(ddd[(ddd['NUSER'] == c[0]) & (ddd['LOAD'] == c[1]) & (ddd['SR'] == c[2])]) == 3:
        return False
    return True


def check_works(c):
    for rep in range(20):
        confs_list = os.listdir("sockshop/works/work_rep" + str(rep) + "/generated_configs")
        for conf_read in confs_list:
            with open("sockshop/works/work_rep" + str(rep) + "/generated_configs/" + conf_read, 'r') as f_conf:
                cc = json.load(f_conf)

                if cc['nusers'][0] == c[0] and cc['loads'][0] == c[1] and cc['spawn_rates'][0] == c[2]:
                    return False
    return True


def check_works_mlp(c):
    for rep in range(20):
        confs_list = os.listdir("sockshop/works_mlp/mlp_predictor_rep" + str(rep) + "/generated_configs")
        for conf_read in confs_list:
            with open("sockshop/works_mlp/mlp_predictor_rep" + str(rep) + "/generated_configs/" + conf_read,
                      'r') as f_conf:
                cc = json.load(f_conf)

                if cc['nusers'][0] == c[0] and cc['loads'][0] == c[1] and cc['spawn_rates'][0] == c[2]:
                    return False
    return True


def check_works_random(c):
    for rep in range(20):
        confs_list = os.listdir("sockshop/works_random/random_predictor_rep" + str(rep) + "/generated_configs")
        for conf_read in confs_list:
            with open("sockshop/works_random/random_predictor_rep" + str(rep) + "/generated_configs/" + conf_read,
                      'r') as f_conf:
                cc = json.load(f_conf)

                if cc['nusers'][0] == c[0] and cc['loads'][0] == c[1] and cc['spawn_rates'][0] == c[2]:
                    return False
    return True


configs_compatte = []

paths = ["sockshop/works/work_rep",
         "sockshop/works_mlp/mlp_predictor_rep",
         "sockshop/works_random/random_predictor_rep",
         "sockshop/NO_ANOMALY/works_causal/rep",
         "sockshop/NO_ANOMALY/works_mlp/rep"]
path_dir_out = "sockshop/compact_confs_final_20rep"

reps = range(20)
count = 0
for path in paths:
    for rep in reps:

        path_act = path + str(rep) + "/generated_configs"
        if not os.path.exists(path_act):
            print("saltato: " + path_act)
            continue
        configs = os.listdir(path_act)

        for conf in configs:
            with open(path_act + "/" + conf, 'r') as f_conf:
                jconf = json.load(f_conf)
                tripla = (jconf['nusers'][0], jconf['loads'][0], jconf['spawn_rates'][0])
                count += 1
                if tripla[0] < 4:
                    print(path_act + "/" + conf)

                if tripla not in configs_compatte:
                    # if check(tripla) and check_works(tripla) and check_works_mlp(tripla) and check_works_random(tripla):
                    if check(tripla):
                        configs_compatte.append(tripla)

print(configs_compatte)
print("{}/{}".format(len(configs_compatte), count))
print("Estimated Time: {}h".format(len(configs_compatte) * 15 / 60))

os.mkdir(path_dir_out)
os.mkdir(path_dir_out + "/configs")

for n, conf in enumerate(configs_compatte):
    with open(path_dir_out + "/configs/conf" + str(n + start_num) + ".json", 'w') as f_conf:
        json.dump({'nusers': [conf[0]], 'loads': [conf[1]], 'spawn_rates': [conf[2]]}, f_conf)
