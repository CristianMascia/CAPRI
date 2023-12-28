import lingam
import numpy as np
from dowhy import gcm
import utils
import data_preparation


def build_model(df, path_dag, prior_knowledge=None):
    X = df.to_numpy(dtype=np.float64)
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)
    adj_mat = utils.threshold_matrix(np.transpose(model.adjacency_matrix_),
                                     0.3)  # TODO: aggiornare con il nuovo valore o modello

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
    df = data_preparation.read_experiments(path_data, {s: s for s in services})
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

    #TODO: aggiungere generazione configurazioni
    quit()

# muBench_example()
