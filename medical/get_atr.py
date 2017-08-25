from tools.wfdbtools import rdann

def get_true_r_waves(path_string, start_sample, stop_sample):

    result_qrs = rdann(path_string, 'qrs', start=0, end=stop_sample, types=[])
    r_waves = []

    for element in result_qrs:
        if start_sample <= element[0] <= stop_sample:
            r_waves.append(element[0])
        else:
            break

    return r_waves