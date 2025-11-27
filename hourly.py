from physf import *
from dataf import *

TIMESHIFT = 0
SIZE = 6
BIN = 0.1  # seconds


def hour_by_hour(x_list: list):

    XTYPE = input('Global (\'global\') or local (\'local\') stimulation? ')

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
            df = extract_opto(PATH, TIMESHIFT)

        else:
            raise ValueError(
                'The stimulation type needs to be global or local.')

        corrs = construct_corr_matrix(df)

        subsystem, sub_corrs = filter_neurons(corrs, 60, SIZE)

        print(f'NEURONS CHOSEN:\t{subsystem}.')
        print(f'CORRELATIONS:\t{sub_corrs}.\n')

        _, stim, net = select_neuron_subset(df, subsystem)

        hours = np.arange(0.5, 20.5, 1)

        for hour in hours:
            
            t = np.arange(0, 720000).T # binned time for splitting up hour-by-hour

            cond = np.where(np.abs(t - hour*36000) <= 18000)
            mask = np.isin(t, cond)

            sub_stim = stim[mask]
            sub_net = net[mask, :]

            j = maxent(sub_stim, sub_net, SIZE)
            ba, pa, _, _ = cm_metrics(sub_stim, sub_net, j)

            print(ba, pa)

            bx.append(ba)
            px.append(pa)

        b.append(bx)
        p.append(px)
    
    b, p = np.array(b), np.array(p)

    np.save(f'{XTYPE}/{XTYPE} hh baseline.npy', b)
    np.save(f'{XTYPE}/{XTYPE} hh model.npy', p)

if __name__ == '__main__':
    xl = [9, 12, 14, 15, 16, 17, 18, 19]
    xg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # hour_by_hour(xl)
    hour_by_hour(xg) # need to check if i'm assigning the stimuli incorrectly


# testT = np.arange(0.5, 20.5, 1)
# testR = np.random.normal(0, 1, 40).reshape((2, 20))
# cond = np.where(np.abs(testT - 15 / 2) <= 5 / 2)
# mask = np.isin(testT, cond)

# print(mask)

# print(f'Original array: {testR}\n')
# print(f'Filtered array: {testR[:, mask]}')
