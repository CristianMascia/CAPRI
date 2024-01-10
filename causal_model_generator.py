import json
import os.path

import lingam
import numpy as np
from dowhy import gcm
import utils
import data_preparation
from configuration_generator import generate_config


def build_model(df, path_dag, prior_knowledge=None):
    X = df.to_numpy(dtype=np.float64)
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)
    adj_mat = utils.threshold_matrix(np.transpose(model.adjacency_matrix_), 0)

    causal_graph = utils.adjmat2dot(adj_mat, df.columns)
    utils.save_dot(causal_graph, path_dag)
    causal_model = gcm.StructuralCausalModel(causal_graph)
    gcm.auto.assign_causal_mechanisms(causal_model, df, quality=gcm.auto.AssignmentQuality.GOOD)
    gcm.fit(causal_model, df)
    # TODO: salvare causal model
    return causal_model


def muBench_example():
    services = ['s' + str(i) for i in range(10)]
    path_data = "muBench/data"
    path_wm = "muBench/configs/workmodel.json"
    path_df = "muBench_df.csv"

    print("------READING EXPERIMENTS------")
    df = data_preparation.read_experiments(path_data, {s: s for s in services}, services,
                                           lambda p: p[:p.rfind("-", 0, p.rfind("-"))])
    df.to_csv(path_df, index=False)

    print("------SPLITTING DATASET------")

    nusers = list(set(df['NUSER']))
    nusers_discovery = [nusers[i] for i in [0] + [i for i in range(2, len(nusers) - 1, 2)] + [len(nusers) - 1]]
    print("Using NUSERs: " + str(nusers_discovery))
    df_discovery = utils.hot_encode_col(df[df['NUSER'].isin(nusers_discovery)].reset_index(), 'LOAD')

    print("------GENERATING PRIOR KNOWLEDGE------")
    pk = utils.get_generic_priorknorledge_mat(df_discovery.columns, services, path_wm)

    print("------CAUSAL DISCOVERY------")
    causal_model = build_model(df_discovery, "mubench_example", pk)

    # TODO: aggiungere generazione configurazioni
    quit()


def sockshop_example():
    with open("sockshop/pods.txt", 'r') as f_pods:
        pods = [p.replace("\n", "") for p in f_pods.readlines()]

        with open("sockshop/mapping_service_request.json", "r") as f_map:
            mapping = json.load(f_map)
            services = list(mapping.keys())

            path_df = "sockshop/data/sockshop_df.csv"
            path_configs = "sockshop/configs"

            print("------READING EXPERIMENTS------")
            df = data_preparation.read_experiments("sockshop/data", mapping, pods, data_preparation.rename_startwith)
            df.to_csv(path_df, index=False)

            df_discovery, loads_mapping = utils.hot_encode_col_mapping(df, 'LOAD')

            soglie = {}
            with open("sockshop/soglie.json", 'w') as f_soglie:
                for ser in services:
                    soglie[ser] = utils.calc_thresholds(df, ser)
                json.dump(soglie, f_soglie)

            print("------GENERATING PRIOR KNOWLEDGE------")
            pk = utils.get_generic_priorknorledge_mat(df_discovery.columns, services)
            print("------CAUSAL DISCOVERY------")
            causal_model = build_model(df_discovery, "sockshop_example", pk)

            print("------GENERATING CONFIGRATIONS-")
            for ser in services:
                print(ser)
                for met in ['RES_TIME', 'CPU', 'MEM']:
                    print(met)
                    generate_config(causal_model, df_discovery, ser,
                                    os.path.join(path_configs, "{}_{}_{}.json".format("configs", met, ser)),
                                    loads_mapping,
                                    metrics=[met],
                                    stability=0, nuser_limit=50)
                generate_config(causal_model, df_discovery, ser,
                                os.path.join(path_configs, "{}_{}_{}.json".format("configs", "all", ser)),
                                loads_mapping,
                                metrics=None,
                                stability=0, nuser_limit=50)


def trainticket_example():
    with open("trainticket/pods.txt", 'r') as f_pods:
        pods = [p.replace("\n", "") for p in f_pods.readlines()]

        with open("trainticket/mapping_service_request.json", "r") as f_map:
            mapping = json.load(f_map)
            services = list(mapping.keys())

            path_df = "trainticket/data/trainticket_df.csv"
            path_configs = "trainticket/configs"

            print("------READING EXPERIMENTS------")
            df = data_preparation.read_experiments("trainticket/data", mapping, pods, data_preparation.rename_startwith)
            df.to_csv(path_df, index=False)

            df_discovery, loads_mapping = utils.hot_encode_col_mapping(df, 'LOAD')

            soglie = {}
            with open("trainticket/soglie.json", 'w') as f_soglie:
                for ser in services:
                    soglie[ser] = utils.calc_thresholds(df, ser)
                json.dump(soglie, f_soglie)

            print("------GENERATING PRIOR KNOWLEDGE------")
            pk = utils.get_generic_priorknorledge_mat(df_discovery.columns, services)
            print("------CAUSAL DISCOVERY------")
            causal_model = build_model(df_discovery, "trainticket_example", pk)

            print("------GENERATING CONFIGRATIONS-")
            for ser in services:
                print(ser)
                for met in ['RES_TIME', 'CPU', 'MEM']:
                    print(met)
                    generate_config(causal_model, df_discovery, ser,
                                    os.path.join(path_configs, "{}_{}_{}.json".format("configs", met, ser)),
                                    loads_mapping,
                                    metrics=[met],
                                    stability=0, nuser_limit=50)
                generate_config(causal_model, df_discovery, ser,
                                os.path.join(path_configs, "{}_{}_{}.json".format("configs", "all", ser)),
                                loads_mapping,
                                metrics=None,
                                stability=0, nuser_limit=50)

# muBench_example()


# trainticket_example()
