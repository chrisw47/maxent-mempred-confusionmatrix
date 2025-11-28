import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import re
import h5py
from physf import likelihood

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": "Computer Modern"
})

BIN = 0.1  # (sec)


def load_data(path):
    '''
    Takes a MATLAB file whose format is consistent with those found in the [Dryad data](https://doi.org/10.5061/dryad.p5hqbzkqj), opens and returns time and signals as arrays.

    Args
    ----
    path : str
        Path to the file that needs to be opened.

    Returns
    -------
    Time : np.ndarray
        The time array corresponding to when signals were recorded, in seconds.
    Signals : np.ndarray
        The electrode at which the signal was recorded.
    '''

    data = sio.loadmat(path)['data']
    rate = int(data['sampleRate'][0][0])

    return data[0, 0]['Ts'] / rate, data[0, 0]['Cs']


def split_array(neuron_array: np.ndarray, TIMESHIFT: float, BINSIZE: float):
    '''
    Splits the neuron array into a time series dataframe, with the column 'Neuron 60' corresponding to the stimulus. `neuron_array` is assumed to have been processed such that the last column is the stimulus.

    **NOTE: this function takes into account the timeshift and assumes rows are time bins.**
    '''
    msg = f'SPLITTING ARRAY ACCORDING TO TIMESHIFT. CURRENT TIMESHIFT TO SPLIT: {TIMESHIFT * BINSIZE} s.'
    print(msg)
    print('=' * len(msg) + '\n')

    # stimulus is assumed to be the last neuron in the array (neuron 60 for local cases)
    stim, net = neuron_array[:, -1], neuron_array[:, :-1]

    if TIMESHIFT > 0:
        stim, net = np.atleast_2d(stim[TIMESHIFT:]).T, net[:-TIMESHIFT, :]
    elif TIMESHIFT < 0:
        stim, net = np.atleast_2d(stim[:TIMESHIFT]).T, net[-TIMESHIFT:, :]
    else:
        stim = np.atleast_2d(stim).T

    timeshifted_data = np.hstack((net, stim))

    df = pd.DataFrame(timeshifted_data, columns=[
        f'Neuron {i}' for i in range(61)])

    return df  # time bins are rows


def neuron_system(times, signals, datasize):
    '''
    Construct a pandas dataframe from the times and signals of experimental data. This will be used to construct a correlation matrix.
    '''

    neuron_array = np.zeros((datasize, 61))  # 60 neurons and stimulus
    for time, signal in zip(times, signals):
        # times is floored to remove decimals
        row, col = int(np.floor(time)), int(signal)
        if row >= datasize:
            pass
        else:
            neuron_array[row, col] = 1

    df = pd.DataFrame(neuron_array, columns=[f'Neuron {i}' for i in range(61)])

    return df, neuron_array  # neuron_array in time series form, consisting of all neurons


def construct_corr_matrix(data: pd.DataFrame):
    '''
    Constructs correlation matrix for the provided time series dataframe, here denoted as `data`.
    '''
    return data.corr(method='pearson')


def plot_correlation_matrix(corrs: pd.DataFrame, stim_neuron: int, identifier: int, savefig: bool = False):
    '''
    Plots a correlation matrix of neuron array (as df).

    Args
    ----
    corrs : pd.DataFrame
        Correlation matrix for neurons.
    stim_neuron : int
        Number corresponding to stimulus neuron. For my cases, I set it to 60.
    identifier : int
        Experiment number corresponding to MATLAB file.
    savefig : bool
        Whether to save the figure. By default, it is set to False.
    '''

    _, ax = plt.subplots(figsize=(20, 3))
    data = np.atleast_2d(corrs[f'Neuron {stim_neuron}'].to_numpy())
    im = ax.imshow(data, cmap='coolwarm', aspect='auto')
    ax.set_yticks([0])
    ax.set_yticklabels([f'Neuron {stim_neuron}'])
    ax.set_xticks(range(len(data[0])))
    ax.set_xticklabels([f'N{i}' for i in range(len(data[0]))], rotation=45)
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.title(
        f'Correlation matrix of neuron {stim_neuron} in experiment {identifier}.')
    plt.tight_layout()

    if savefig:
        # save fig
        plt.savefig(
            f'correlation matrices/corr matrix exp {identifier}.png', dpi=400)

    plt.show()


def extract_opto(path: str, ts: int, bin: float):
    '''
    Opens the optogenetic stimulation files from [Dryad data](https://doi.org/10.5061/dryad.p5hqbzkqj) and returns dataframe for analysis.

    Args
    ----
    path : str
        The MATLAB file path.
    ts : int
        The timeshift (in unit shifts) with which to modify the time series.
    bin : float
        The time bin/resolution, in seconds, of the system.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with each column corresponding to a neuron and row corresponding to a time bin.
    '''

    # experiment 1 is the only experiment that's written in MATLAB v7.3, others are MATLAB v5.0
    if re.search(r'(\d+)', path).group() == '1':
        f = h5py.File(path)
        data = f['data']
        rate = int(data['sampleRate'][0][0])
        # convert all times to seconds
        t, c, stim = data['Ts'][0] / rate, data['Cs'][0], data['StimTimes'][0]

    else:
        f = sio.loadmat(path)  # for all other experiments
        data = f['data'][0, 0]
        rate = int(data['sampleRate'][0][0])
        # convert all times to seconds
        t, c, stim = data['Ts'] / rate, data['Cs'], data['StimTimes']

    sig = np.full_like(stim, 60)

    t = np.concatenate((t, stim))
    s = np.concatenate((c, sig))

    # everything gets sorted temporally here
    _, system = neuron_system(t / bin, s, int(72000 / bin))

    df = split_array(system, ts, bin)

    return df


def filter_neurons(corrs: pd.DataFrame, stim_neuron: int, system_size: int) -> tuple:
    '''
    Filters neuron array from original size to a smaller system size for MaxEnt fitting.

    Args
    ----
    corrs : pd.DataFrame
        The correlation matrix of the whole system.
    stim_neuron : int
        The neuron number corresponding to the stimulus (default: 60).
    system_size : int
        How many neurons, including stimulus, go into the MaxEnt fitting process.

    Returns
    -------
    selected_neurons : list 
        List of neuron indices most correlated with stimulus (length: system_size-1).
    correlations : list
        List of correlation values corresponding to selected neurons.
    '''

    # Get correlations with stimulus neuron, selects the top system_size neurons, including stimulus.
    stim_corrs = corrs[f'Neuron {stim_neuron}'].sort_values(
        ascending=True).dropna()[-system_size:]

    # Creates neuron shortlist and associated correlations
    neuron_shortlist, correlation_shortlist = stim_corrs.keys(
    ).to_list(), stim_corrs.round(2).to_list()

    # Converts selected neurons into a list of indices
    selected_neurons = [int(selected_neuron[7:])
                        for selected_neuron in neuron_shortlist]

    return selected_neurons, correlation_shortlist


def select_neuron_subset(full_neuron_array: pd.DataFrame, subset_list: list):
    '''
    From the full neuron time series array, selects a subset of the optimal neurons by correlation and returns a time series containing only those neurons.

    Parameters
    ----------
    full_neuron_array : pd.DataFrame
        The full neuron array containing all the neurons in the system
    subset_list : list
        An optimal subset derived by Pearson correlation via the correlation matrix of all neurons in the system

    Returns
    -------
    neuron_subset_timeseries : np.ndarray
        A time series corresponding to the chosen subset of neurons with time as rows (s-by-n array for system size s).
    stim : np.ndarray
        A time series of the stimulus neuron (1-by-n array).
    net : np.ndarray
        A time series corresponding to the rest of the subset timeseries, aka. the network timeseries ((s-1)-by-n array).
    '''

    neuron_subset_timeseries = full_neuron_array[[
        f'Neuron {i}' for i in subset_list]]

    nst = neuron_subset_timeseries.to_numpy()
    stim, net = nst[:, -1], nst[:, :-1]

    return nst, stim, net


def conf_int(array: np.ndarray, array_mean: np.ndarray, percent: int):
    '''
    Constructs a confidence interval for an array of data that can be used when plotting errorbars in matplotlib.

    Parameters
    ----------
    array : numpy.ndarray
        Data array from which to construct confidence interval
    array_mean : numpy.ndarray
        The mean of the array
    percent : int
        The CI percent

    Returns
    -------
    conf_int : np.ndarray
        The confidence interval expressed as a lower and upper bound in the form of an array
    '''

    conf_int = np.array([np.abs(np.percentile(array, (50-percent / 2), axis=0)-array_mean),
                        np.abs(np.percentile(array, (50+percent / 2), axis=0)-array_mean)])

    return conf_int


def mean_and_ci(a: np.ndarray, pct: int):
    '''
    Given an array in the form generated by `main_analysis` from `gen cm metrics.py`, returns the experiment-averaged mean and confidence interval.

    Args
    ----
    a : np.ndarray
        The array to be used. Specfications are denoted in the summary above.
    pct: int
        The confidence interval percent.

    Returns
    -------
    mean: np.ndarray
        The experiment-averaged performance metrics of baseline/model.
    ci: np.ndarray
        The confidence interval for the mean array.
    '''

    mean = np.mean(a, axis=0)
    ci = conf_int(a, mean, pct)

    return mean, ci


def cm_metrics(stimulus, network, op_params) -> tuple[float]:
    """
    Compute classification performance metrics for a maximum entropy model. Assuming fixed timeshift.

    Parameters
    ----------
    stimulus : numpy.ndarray
        Binary stimulus values (0 or 1)
    network : numpy.ndarray
        Network neuron states, shape (num_neurons, num_samples)
    op_params : numpy.ndarray
        Optimized coupling parameters (J matrix)

    Returns
    -------
    baseline_accuracy : float
        Fraction of samples with stimulus = 0
    predictive_accuracy : float
        Overall classification accuracy
    true_negative_rate : float
        Normalized true negative rate
    true_positive_rate : float
        Normalized true positive rate

    Notes
    -----
    Classifies stimulus using likelihood ratio > 1 for stimulus = 1.

    **NOTE: time bins are in rows.**
    """
    print('GENERATING CONFUSION MATRIX METRICS...\n' + '=' * 38 + '\n')

    confusion_matrix = np.zeros((2, 2))  # initialize confusion matrix

    for i in range(len(stimulus)):
        OUTPUT = 0.0

        stim = stimulus[i]
        net_vec = network[i, :]

        likelihood_ratio = likelihood(op_params, net_vec)

        if likelihood_ratio > 1:
            OUTPUT = 1.0

        if stim == 0 and OUTPUT == 0:
            confusion_matrix[0, 0] += 1
        elif stim == 1 and OUTPUT == 0:
            confusion_matrix[0, 1] += 1
        elif stim == 1 and OUTPUT == 1:
            confusion_matrix[1, 1] += 1
        elif stim == 0 and OUTPUT == 1:
            confusion_matrix[1, 0] += 1

    ACTUAL_0 = confusion_matrix[0, 0] + confusion_matrix[1, 0]
    ACTUAL_1 = confusion_matrix[0, 1] + confusion_matrix[1, 1]

    # DEBUG
    print(f'Confusion matrix:\n{confusion_matrix}')
    # DEBUG

    total_elts = np.sum(confusion_matrix)

    BASELINE_ACCURACY = ACTUAL_0 / total_elts
    PREDICTIVE_ACCURACY = np.trace(confusion_matrix) / total_elts

    # normalizes the confusion matrix
    confusion_matrix[0, 0] /= ACTUAL_0
    confusion_matrix[1, 0] /= ACTUAL_0
    confusion_matrix[0, 1] /= ACTUAL_1
    confusion_matrix[1, 1] /= ACTUAL_1

    baseline_accuracy = float(round(BASELINE_ACCURACY, 6))
    predictive_accuracy = float(round(PREDICTIVE_ACCURACY, 6))
    true_negative_rate = float(round(confusion_matrix[0, 0], 6))
    true_positive_rate = float(round(confusion_matrix[1, 1], 6))

    print(
        f'Baseline accuracy: {baseline_accuracy},\t Predictive accuracy: {predictive_accuracy}')
    print(
        f'True negative rate: {true_negative_rate},\t True positive rate: {true_positive_rate}\n\n')
    print(f'Normalized confusion matrix: \n{confusion_matrix}' + '\n' * 5)

    return baseline_accuracy, predictive_accuracy, true_negative_rate, true_positive_rate
