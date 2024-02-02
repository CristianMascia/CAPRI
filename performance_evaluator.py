import glob
import json
import os
import string
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import data_preparation
import utils


def calc_hamming_distance(conf_real, conf_pred):
    return abs(conf_real['NUSER'] - conf_pred['nusers'][0]) / 3 + (
        0 if conf_real['LOAD'] == conf_pred['loads'][0] else 1) / 3 + (
        0 if conf_real['SR'] == conf_pred['spawn_rates'][0] else 1) / 3


def calc_metrics(df: Union[str, pd.DataFrame], get_config: Callable[[str, str], Tuple[dict, float]], services,
                 model_limit,
                 ths_filtered=False, metrics=None, sensibility=0.):
    if metrics is None:
        metrics = ['RES_TIME', 'CPU', 'MEM']

    if isinstance(df, str):
        df = pd.read_csv(df)
    elif not isinstance(df, pd.DataFrame):
        raise ValueError("The parameter df must be a path or a DataFrame.")

    def calc_mhd(c_pred, true_positive=True):

        if true_positive:
            nusers = list(set(df['NUSER']) & set(range(1, c_pred['nusers'][0])))
        else:
            nusers = list(set(df['NUSER']) & set(range(c_pred['nusers'][0], model_limit + 1)))

        loads = list(set(df['LOAD']))
        srs = list(set(df['SR']))
        nusers.sort()
        for n1 in nusers:
            dists = []
            for l1 in loads:
                for sr1 in srs:
                    value1 = df[(df['NUSER'] == n1) & (df['LOAD'] == l1) & (df['SR'] == sr1)][met + "_" + ser].mean()
                    if value1 > ths[met]:
                        dists.append(calc_hamming_distance({'NUSER': n1, 'LOAD': l1, 'SR': sr1}, c_pred))
            if len(dists) > 0:
                return np.min(dists)
        return 0

    metrics_dict = {}

    for met in metrics:
        mhd_values_pos = []
        mhd_values_false = []
        true_positive = 0
        false_negative = 0
        false_positive = 0
        for ser in services:
            ths = utils.calc_thresholds(df, ser, filtered=ths_filtered)
            config, value_config = get_config(ser, met)
            if config is not None:
                if value_config > ((1 - sensibility) * ths[met]):
                    true_positive += 1
                    mhd_values_pos.append(calc_mhd(config))
                else:
                    false_positive += 1
                    mhd_values_false.append(calc_mhd(config, False))

            else:
                # print("No anomaly for {} for {}".format(ser, met))
                for n in reversed(range(1 + max(df['NUSER']))):
                    for l in list(set(df['LOAD'])):
                        for sr in list(set(df['SR'])):
                            a = df[(df['NUSER'] == n) & (df['LOAD'] == l) & (df['SR'] == sr)]
                            if a[met + "_" + ser].mean() > ((1 + sensibility) * ths[met]):
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

        if len(mhd_values_pos) > 0:
            mhd_pos = np.mean(mhd_values_pos)
        else:
            mhd_pos = -1

        if len(mhd_values_false) > 0:
            mhd_false = np.mean(mhd_values_false)
        else:
            mhd_false = -1

        metrics_dict[met] = {"precision": precision, "recall": recall, "mhd_pos": mhd_pos, "mhd_false": mhd_false}

    return metrics_dict
