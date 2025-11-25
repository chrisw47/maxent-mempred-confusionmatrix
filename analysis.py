from physf import *
from dataf import *
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": "Computer Modern"
})

BIN = 0.1  # in seconds, determines the time binning


def main_analysis(experiment_list: list, timeshifts: np.ndarray, system_sizes: np.ndarray):
    '''
    Runs main analysis of experiment to generate confusion matrix metrics for real data.

    Args
    ----
    experiment_list : list
        A list of experiment numbers that are used to index files from [Dryad data](https://datadryad.org/dataset/doi:10.5061/dryad.p5hqbzkqj).
    timeshifts: np.ndarray
        An array that lists timeshifts (in unit shifts) to run the MaxEnt model. For instance, if `BIN=0.1` (s), and `timeshifts=[-1 0 1]`, then the model is evaluated at -0.1, 0, and 0.1 s timeshift. Refer to Fig 1 in [this paper](https://academic.oup.com/pnasnexus/article/2/6/pgad188/7202378) for context on timeshifting.
    system_sizes: np.ndarray
        An array that tells the model how many neurons are used to determine the *J* matrix.

    From here, a few arrays are generated and saved as `.npy` files. Notably, the predictive accuracy, true positive rate, and true negative rate of both baseline and model are saved, resulting in six total files.
    '''

    b = []  # predictive accuracy
    p = []
    b_tp = []  # true positives
    p_tp = []
    b_tn = []  # true negatives
    p_tn = []
    x_type = input(f'Global (\'global\') or local (\'local\') stim? ')

    for num in experiment_list:

        if x_type == 'local':
            PATH = f'real data/experiment_{num}_20h_stimL.mat'
            Ts, Cs = load_data(PATH)
            _, full_array = neuron_system(
                Ts / BIN, Cs, int(72000 / BIN))

        elif x_type == 'global':
            PATH = f'real data/experiment_{num}_20h_stim.mat'

        else:
            raise ValueError(
                'The stimulation type needs to be global or local.')

        b_ts = []
        p_ts = []
        b_tp_ts = []
        p_tp_ts = []
        b_tn_ts = []
        p_tn_ts = []

        for ts in timeshifts:
            if x_type == 'local':
                df = split_array(full_array, ts, BIN)

            elif x_type == 'global':
                df = extract_opto(PATH, ts)

            correlations = construct_corr_matrix(df)
            # plot_correlation_matrix(correlations, 60, num)

            b_ss = []
            p_ss = []
            b_tp_ss = []
            p_tp_ss = []
            b_tn_ss = []
            p_tn_ss = []

            for ss in system_sizes:
                print(
                    f'BEGINNING ANALYSIS FOR EXPERIMENT {num}. SYSTEM SIZE: {ss} NEURONS. TIMESHIFT: {ts * BIN} S.')

                neuron_candidates, candidate_corrs = filter_neurons(
                    correlations, 60, ss)

                print(f'NEURONS CHOSEN:\t{neuron_candidates}.')
                print(f'CORRELATIONS:\t{candidate_corrs}.\n')

                _, ts_stim, ts_net = select_neuron_subset(
                    df, neuron_candidates)  # time bins are rows

                j = maxent(ts_stim, ts_net, ss)

                ba, pa, tn, tp = cm_metrics(ts_stim, ts_net, j)

                b_ss.append(ba)
                p_ss.append(pa)
                b_tp_ss.append(0)
                p_tp_ss.append(tp)
                b_tn_ss.append(1)
                p_tn_ss.append(tn)

            b_ts.append(b_ss)
            p_ts.append(p_ss)
            b_tp_ts.append(b_tp_ss)
            p_tp_ts.append(p_tp_ss)
            b_tn_ts.append(b_tn_ss)
            p_tn_ts.append(p_tn_ss)

        b.append(b_ts)
        p.append(p_ts)
        b_tp.append(b_tp_ts)
        p_tp.append(p_tp_ts)
        b_tn.append(b_tn_ts)
        p_tn.append(p_tn_ts)

    arrs = [b, p, b_tp, p_tp, b_tn, p_tn]

    for a, name in zip(arrs, ['b', 'p', 'b_tp', 'p_tp', 'b_tn', 'p_tn']):
        a = np.array(a)
        np.save(f'{x_type}/{x_type} stim {name}.npy', a)

    return b, p, b_tp, p_tp, b_tn, p_tn


if __name__ == '__main__':
    experiment_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # [9, 12, 14, 15, 16, 17, 18, 19]
    timeshifts = np.arange(-2, 3)
    system_sizes = np.arange(3, 7)

    b, p, b_tp, p_tp, b_tn, p_tn = main_analysis(
        experiment_nums, timeshifts, system_sizes)

    print(f'BASELINE PREDICTIVE ACCURACY:\n{b}\n\n')
    print(f'MODEL PREDICTIVE ACCURACY:\n{p}\n\n')
    print(f'BASELINE TRUE POSITIVES:\n{b_tp}\n\n')
    print(f'MODEL TRUE POSITIVES:\n{p_tp}\n\n')
    print(f'BASELINE TRUE NEGATIVES:\n{b_tn}\n\n')
    print(f'MODEL TRUE NEGATIVES:\n{p_tn}\n\n')
