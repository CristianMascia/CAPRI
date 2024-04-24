import json
import os

configs_compatte = []

paths = ["onlineboutique/works/work_rep",
         "onlineboutique/works_mlp/mlp_predictor_rep",
         "onlineboutique/works_random/random_predictor_rep"]
path_dir_out = "onlineboutique/compact_confs_first_10rep"

reps = range(10)
count = 0
for path in paths:
    for rep in reps:

        path_act = path + str(rep) + "/generated_configs"
        configs = os.listdir(path_act)

        for conf in configs:
            with open(path_act + "/" + conf, 'r') as f_conf:
                jconf = json.load(f_conf)
                tripla = (jconf['nusers'][0], jconf['loads'][0], jconf['spawn_rates'][0])
                count += 1
                if tripla not in configs_compatte:
                    configs_compatte.append(tripla)

print(configs_compatte)
print("{}/{}".format(len(configs_compatte), count))

os.mkdir(path_dir_out)
os.mkdir(path_dir_out + "/configs")

for n, conf in enumerate(configs_compatte):
    with open(path_dir_out + "/configs/conf" + str(n) + ".json", 'w') as f_conf:
        json.dump({'nusers': [conf[0]], 'loads': [conf[1]], 'spawn_rates': [conf[2]]}, f_conf)
