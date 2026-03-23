from physf import *
from dataf import *

plt.rcParams.update({
    "font.size": 20,
    "legend.fontsize": 16
})

TIMESHIFT = 0
SIZE = 6
BIN = 0.1  # seconds


def hour_by_hour(x_list: list, XTYPE: str):

    b, p = [], []

    for x in x_list:

        bx, px = [], []

        print(f'BEGINNING ANALYSIS ON EXPERIMENT {x}.')

        if XTYPE == 'local':
            PATH = f'real data/exp{x}_local.mat'
            t, c = load_data(PATH)
            _, full = neuron_system(t / BIN, c, int(72000 / BIN))
            df = split_array(full, TIMESHIFT, BIN)

        elif XTYPE == 'global':
            PATH = f'real data/experiment_{x}_20h_stim.mat'
            df = extract_opto(PATH, TIMESHIFT, BIN)

        else:
            raise ValueError(
                'The stimulation type needs to be global or local.')

        # corrs = construct_corr_matrix(df)

        # subsystem, sub_corrs = filter_neurons(corrs, 60, SIZE)

        # print(f'NEURONS CHOSEN:\t{subsystem}.')
        # print(f'CORRELATIONS:\t{sub_corrs}.\n')
        hours = np.arange(0.5, 20.5, 1)
        system_sizes = [2,3,4,5,6]

        for hour in hours:

            # binned time for splitting up hour-by-hour
            t = np.arange(0, 720000).T

            cond = np.where(np.abs(t - hour*36000) <= 18000)
            mask = np.isin(t, cond)
            subset = set([60])

            for s in system_sizes:
                neurons_to_try = set(range(1, 61)) - subset

                print(f'Neurons to try:\t{neurons_to_try}.')

                # list of predictive accuracies to document which neuron adds most beneficially to overall predictive accuracy.
                metrics = []

                for n in neurons_to_try:
                    subset.add(n)

                    print(f'Trying subset containing neurons:\t {sorted(subset)}.')

                    _, ts_stim, ts_net = select_neuron_subset(
                        df, sorted(subset))  # time bins are rows, ts_stim: timeshifted stimulus array, ts_net: timeshifted network array. this selection applies when we choose off of optimal contribution to predictive accuracy.
                    
                    sub_stim, sub_net = ts_stim[mask], ts_net[mask, :]

                    j = maxent(sub_stim, sub_net, s)

                    ba, pa, mn, mp = cm_metrics(sub_stim, sub_net, j)

                    metrics += [n, ba, pa, mn, mp]
                    subset.remove(n)

                    print(
                        f'List of neuron indices and predictive accuracies:\n\n{metrics}')

                metrics = np.array(metrics).reshape(
                    (len(neurons_to_try), 5))  # reshapes into an nx5 array

                # Find the neuron that maximizes model predictive accuracy
                max_idx = np.argmax(metrics[:, 2])
                best_neuron_idx = int(metrics[max_idx, 0])
                subset.add(best_neuron_idx)

                ba, pa, mn, mp = metrics[max_idx, 1:]

                if s == 6:
                    bx.append(ba)
                    px.append(pa)
                
        b.append(bx)
        p.append(px)

    b, p = np.array(b), np.array(p)

    np.save(f'{XTYPE}/{XTYPE} hh baseline revised.npy', b)
    np.save(f'{XTYPE}/{XTYPE} hh model revised.npy', p)


if __name__ == '__main__':
    # # xl = [9, 12, 14, 15, 16, 17, 18, 19]
    # xg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # # hour_by_hour(xl, 'local')
    # hour_by_hour(xg, 'global')
    # quit()
    blhh = np.load('local/local hh baseline revised.npy')  # baseline local hour-hour
    plhh = np.load('local/local hh model revised.npy')
    bghh = np.load('global/global hh baseline revised.npy')
    pghh = np.load('global/global hh model revised.npy')

    # print(f'Baseline global:\n\n{bghh}\n\nModel global:\n\n{pghh}')

    pghhm, pghhci = mean_and_ci(pghh, 95)
    plhhm, plhhci = mean_and_ci(plhh, 95)

    bghhm, bghhci = mean_and_ci(bghh, 95)
    blhhm, blhhci = mean_and_ci(blhh, 95)

    bhhm = np.mean(np.concatenate((np.atleast_2d(bghhm), np.atleast_2d(blhhm)), axis=0), axis=0)
    bhhci = np.mean(np.concatenate((np.atleast_2d(bghhci), np.atleast_2d(blhhci)), axis=0), axis=0)

    t = np.arange(1, 21, 1)

    plt.figure(figsize=(12, 7))
    plt.plot(t, bhhm, ms=8, marker='s', c='#D81B60', label='Uncorrelated baseline')
    plt.plot(t, pghhm, ms=8, marker='o', c='#004D40', label='Global stimulation')
    plt.plot(t, plhhm, ms=8, marker='x', c='#1E88E5', label='Local stimulation')
    plt.errorbar(t, bhhm, yerr=bhhci, fmt='none', ecolor='#D81B60', capsize=5, alpha=0.3)
    plt.errorbar(t, pghhm, yerr=pghhci, fmt='none', ecolor='#004D40', capsize=5, alpha=0.3)
    plt.errorbar(t, plhhm, yerr=plhhci, fmt='none', ecolor='#1E88E5', capsize=5, alpha=0.3)

    plt.xticks([0,5,10,15,20])
    plt.xlim(0, 21)
    plt.xlabel('\\textbf{Experiment progression (hours)}')

    plt.ylim(0.965, 1.002)
    plt.yticks([0.98, 0.99, 1.00])
    plt.ylabel('\\textbf{Predictive accuracy}')

    plt.grid(axis='y', ls='--', alpha=0.8)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(f'figs/hh {TIMESHIFT * 100}ms revised.png', dpi=400)
    plt.show()