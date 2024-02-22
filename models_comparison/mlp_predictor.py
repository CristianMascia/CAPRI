import json
import os

import numpy as np
import pandas as pd

import utils
from models_comparison import CONFIG
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
import questions_utils as qu

num_reps = 20


def __main__(path_main_dir):
    df = pd.read_csv(CONFIG.path_df)

    mets = [{}] * num_reps
    for rep in range(num_reps):
        path_dir = os.path.join(path_main_dir, "mlp_predictor_REP_{}".format(rep))
        path_model = os.path.join(path_dir, "model.h5")
        path_dir_configs = os.path.join(path_main_dir, "mlp_predictor_REP_{}".format(rep), "generated_configs")
        path_metric_file = os.path.join(path_main_dir, "mlp_predictor_REP_{}".format(rep), "metrics.json")

        os.makedirs(path_dir, exist_ok=True)
        os.makedirs(path_dir_configs, exist_ok=True)

        mlp(df, path_model, path_dir_configs, norm=True)
        mets[rep] = qu.calc_metrics(df, path_dir_configs, "configs")

        with open(path_metric_file, 'w') as f:
            json.dump(mets[rep], f)

    with open(os.path.join(path_main_dir, "mean_metrics_mlp_predictor.json"), 'w') as f_mean_mets:
        json.dump(qu.merge_met_dict(mets), f_mean_mets)


def mlp(df, path_model, path_dir_configs, norm=False):
    df_train, loads_mapping = utils.hot_encode_col_mapping(df, 'LOAD')
    df_train = df_train[df_train['NUSER'].isin([i for i in range(1, 30, 2)] + [30])]
    input_col = ['NUSER'] + ['LOAD_' + str(i) for i in range(len(loads_mapping.keys()))] + ['SR']
    output_col = [met + "_" + ser for met in CONFIG.all_metrics for ser in CONFIG.services]
    df_train = df_train[input_col + output_col]
    if norm:
        df_train[output_col] = (df_train[output_col] - df_train[output_col].min()) / (
                df_train[output_col].max() - df_train[output_col].min())

    X_train = df_train[input_col].values
    Y_train = df_train[output_col].values

    loads = loads_mapping.keys()
    spawn_rates = list(set(df['SR']))

    thresholds = {}

    for ser in CONFIG.services:
        thresholds[ser] = utils.calc_thresholds(df_train, ser)

    print("------DEFINING MLP------")
    model = Sequential()
    model.add(Input(shape=(5,), dtype=int))  # settare interi
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=len(output_col), activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    print("------TRAIN------")
    kfold = KFold(n_splits=5, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train)):
        print(f"Fold {fold + 1}:")

        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = Y_train[train_index], Y_train[val_index]

        # Addestramento del modello su X_train_fold, y_train_fold
        model.fit(X_train_fold, y_train_fold, epochs=300, batch_size=32, verbose=0)

        # Valutazione del modello su X_val_fold, y_val_fold
        loss, accuracy = model.evaluate(X_val_fold, y_val_fold)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    model.save(path_model)

    print("------GENERATING CONFIGRATIONS-")

    def search_config():
        for n in range(2, 31):
            print(n)
            for load in loads:
                for sr in spawn_rates:

                    input_data = np.zeros(len(input_col), )
                    input_data[input_col.index('NUSER')] = n
                    input_data[input_col.index('SR')] = sr

                    for li in range(len(loads)):
                        input_data[input_col.index('LOAD_{}'.format(li))] = loads_mapping[load][li]

                    target_pred = model.predict(np.array([input_data]))[:, [output_col.index(target_col)]]

                    if target_pred > thresholds[ser][met]:
                        print("CONFIGURAZIONE TROVATA")
                        with open(os.path.join(path_dir_configs, "{}_{}_{}.json".format("configs", met, ser)),
                                  'w') as f_out:
                            json.dump({"nusers": [n], "loads": [load], "spawn_rates": [sr],
                                       "anomalous_metrics": [met]}, f_out)
                        return
        print("NON TROVATA")

    for ser in CONFIG.services:
        for met in CONFIG.all_metrics:
            target_col = met + "_" + ser
            print("Searching configuration for service: {} for metric: {}".format(ser, met))

            search_config()


__main__("mlp_predictor")
