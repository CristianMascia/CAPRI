import glob
import json
import os
import numpy as np
import pandas as pd
import data_preparation
import utils


def calc_metrics(path_df, path_configs, path_run_configs, services, mapping, ths_filtered=False, metrics=None):
    if metrics is None:
        metrics = ['RES_TIME', 'CPU', 'MEM']

    def get_exp_value(edir, s, pod, m):
        if m in ['CPU', 'MEM']:
            mem_cpu_s = glob.glob(os.path.join(edir, 'mem_cpu_*.txt'))
            values = [0.] * len(mem_cpu_s)
            for k, rep in enumerate(mem_cpu_s):
                values[k] = data_preparation.dockerstats_extractor(rep, [pod], data_preparation.rename_startwith)[
                    m + " AVG"].mean()
            return np.mean(values)
        else:
            stats = glob.glob(os.path.join(edir, 'esec_*.csv_stats.csv'))
            values = [0.] * len(stats)
            for k, rep in enumerate(stats):
                loc = data_preparation.locuststats_extractor(rep)
                values[k] = loc[loc['Name'] == s]['Average Response Time'].mean()
            return np.mean(values)

    metrics_dict = {}

    df = pd.read_csv(path_df)
    for met in metrics:
        true_positive = 0
        false_negative = 0
        false_positive = 0
        for ser in services:
            ths = utils.calc_thresholds(df, ser, filtered=ths_filtered)
            name_config = "configs_{}_{}".format(met, ser)
            path_config = os.path.join(path_configs, name_config + ".json")
            if os.path.isfile(path_config):
                with open(path_config, 'r') as f_c:
                    config = json.load(f_c)
                    exp_dir = os.path.join(path_run_configs, name_config,
                                           "experiments_sr_" + str(config['spawn_rates'][0]),
                                           'users_' + str(config['nusers'][0]), config['loads'][0])

                    if get_exp_value(exp_dir, ser, mapping[ser], met) > ths[met]:
                        true_positive += 1
                    else:
                        false_positive += 1

            else:
                # print("No anomaly for {} for {}".format(ser, met))
                for n in reversed(range(1 + max(df['NUSER']))):
                    for l in list(set(df['LOAD'])):
                        for sr in list(set(df['SR'])):
                            a = df[(df['NUSER'] == n) & (df['LOAD'] == l) & (df['SR'] == sr)]
                            if a[met + "_" + ser].mean() > ths[met]:
                                false_negative += 1
                                break
            # print("Service {} Met: {}".format(ser, met))

        precision = 0
        recall = 0

        if true_positive > 0:
            precision = true_positive / (true_positive + false_positive)

            if false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 1
        metrics_dict[met] = {"precision": precision, "recall": recall}

    return metrics_dict
