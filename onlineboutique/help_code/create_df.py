import json

import data_preparation as dp

with open("mapping_service_request.json", 'r') as f_map:
    mapping = json.load(f_map)
    with open("pods.txt", 'r') as f_pods:
        pods = [p.replace("\n", "") for p in f_pods.readlines()]
        df = dp.read_experiments("data", mapping, pods)
        df.to_csv("data/df.csv", index=False)
