import json
import random
import dowhy.gcm as gcm
import utils


def generate_config(causal_model, df, service, path_config, loads_mapping, nuser_limit, nuser_start=2, n_step=1,
                    metrics=None, stability=0, loads=None, spawn_rates=None, show_comment=False):
    def identity_lambda(number):
        return lambda x: number

    def search_config():
        random.shuffle(combinations)
        for load_level, sr_level in combinations:
            anomalous_metrics = metrics.copy()
            for stab in range(stability + 1):
                # print("CHECK N{}".format(stab))

                dict_int = {'NUSER': lambda y: n + stab, 'SR': lambda y: sr_level}
                for li in range(len(loads_mapping.keys())):
                    dict_int['LOAD_{}'.format(li)] = identity_lambda(loads_mapping[load_level][li])

                samples = gcm.interventional_samples(causal_model, dict_int, num_samples_to_draw=1000)

                for amet in anomalous_metrics.copy():
                    if show_comment:
                        print("for {} Pred {} TH {} with ({},{},{})".format(amet, samples[amet + "_" + service].mean(),
                                                                            ths[amet], n, load_level, sr_level))
                    if samples[amet + "_" + service].mean() <= ths[amet]:
                        anomalous_metrics.remove(amet)
                if len(anomalous_metrics) == 0:
                    break
            if len(anomalous_metrics) > 0:
                with open(path_config, 'w') as f_out:
                    json.dump({"nusers": [n], "loads": [load_level], "spawn_rates": [sr_level],
                               "anomalous_metrics": anomalous_metrics}, f_out)
                    return True
        return False

    if metrics is None:
        metrics = ['RES_TIME', 'CPU', 'MEM']

    if loads is None:
        load_levels = loads_mapping.keys()

    if spawn_rates is None:
        sr_levels = list(set(df['SR']))

    ths = utils.calc_thresholds(df, service)

    combinations = [(load, sr) for load in load_levels for sr in sr_levels]
    n = nuser_start

    while not search_config() and n < nuser_limit:
        n += n_step
