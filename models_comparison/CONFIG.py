import os.path

services = ['s' + str(i) for i in range(10)]
loads = ['uniform', 'randomly_balanced', 'unbalanced_one']
SRs = [1, 5, 10]
treatments = ['NUSER', 'LOAD_0', 'LOAD_1', 'LOAD_2', 'SR']
all_metrics = ['RES_TIME', 'CPU', 'MEM']
mubench_path = "../mubench"
path_df = os.path.join(mubench_path, "data/mubench_df.csv")
path_wm = os.path.join(mubench_path, "configs/workmodel.json")
path_minimal_configs_anomaly = os.path.join(mubench_path, "minimal_configs_anomaly.json")
model_limit = 30

path_dag_gnn = "../DAG_from_GNN"
dag_gnn_epochs = 300
