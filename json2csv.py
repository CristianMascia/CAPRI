import json

header = ['Model', 'Metric', 'Precision', 'Recall', 'MHD']
paths = ["sockshop/works/", "sockshop/works_mlp/", "sockshop/works_random/"]
models = ['Causal', 'MLP', 'Random']

with open("prova.csv", 'w') as file:
    for h, v in enumerate(header):
        file.write(v)
        if h != len(header) - 1:
            file.write(",")
    file.write('\n')

    for mod in range(len(models)):


        with open(paths[mod] + "avg_metrics.json", "r") as f_mets:
            mets = json.load(f_mets)
            for m in ['RES_TIME', 'CPU', 'MEM']:
                file.write(models[mod] + ",")
                file.write(m + ",")
                file.write(str(mets[m]['precision_mean']) + ",")
                file.write(str(mets[m]['recall_mean']) + ",")
                file.write(str(mets[m]['mhd_pos_mean']))
                file.write("\n")
