import json
import dowhy.gcm as gcm
import utils


# funziona solo se il modello non presenta archi entrati nei trattamenti,
# in questo modo si mantengono i loro range. per esempio i load rimangono binari
def get_config_from_sample(samples, loads_mapping):
    # NUSER
    nuser = int(samples['NUSER'].mean())

    # LOAD
    mcount = 0
    icount = None
    name_l = ""
    for k, v in loads_mapping.items():
        count = len(samples[samples['LOAD_' + str(v.index(1))] == 1].values)
        if count > mcount:
            mcount = count
            icount = v
            name_l = k
    #  SR
    sr = int(samples[samples['LOAD_' + str(icount.index(1))] == 1]['SR'].mode().values[0])
    return nuser, name_l, sr


def generate_config(causal_model, df, service, path_config, loads_mapping, metrics=['RES_TIME', 'CPU', 'MEM'],
                    stability=0, nuser_limit=-1):
    ths = utils.calc_thresholds(df, service)

    n = 2
    n_step = 1
    anomalous_metrics = []
    while True:
        # print("NUSER: " + str(n))
        samples = gcm.interventional_samples(causal_model, {'NUSER': lambda y: n}, num_samples_to_draw=10000)

        for met in metrics:
            if samples[met + "_" + service].mean() > ths[met]:
                anomalous_metrics.append(met)
        # TODO: compattare con stability
        if len(anomalous_metrics) > 0:
            # print("THRESHOLD EXCEEDED for: " + str(anomalous_metrics))

            anomalous_metrics_after_check = anomalous_metrics
            for stab in range(1, stability + 1):
                # print("TEST STABILITY " + str(stab))
                samples_stab = gcm.interventional_samples(causal_model, {'NUSER': lambda y: n + stab},
                                                          num_samples_to_draw=10000)
                for met_anom in anomalous_metrics:
                    if samples_stab[met_anom + "_" + service].mean() < ths[met_anom]:
                        anomalous_metrics_after_check.remove(met_anom)
                        if len(anomalous_metrics_after_check) == 0:
                            break

            if len(anomalous_metrics_after_check) > 0:
                # print("THRESHOLD EXCEEDED AFTER STABILITY CHECK for: " + str(anomalous_metrics_after_check))
                samples_filterd = samples
                for anom_met in anomalous_metrics_after_check:  # TODO: controllare che il samples filtrato non risulti vuoto
                    samples_filterd = samples_filterd[samples_filterd[anom_met + "_" + service] > ths[anom_met]]

                nuser, load, sr = get_config_from_sample(samples_filterd, loads_mapping)

                with open(path_config, 'w') as f_out:
                    json.dump(
                        {"nusers": [nuser], "loads": [load], "spawn_rates": [sr],
                         "anomalous_metrics": anomalous_metrics_after_check},
                        f_out)
                break
        if nuser_limit != -1 and n >= nuser_limit:
            break
        n += n_step
