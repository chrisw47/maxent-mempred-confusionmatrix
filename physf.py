import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import time


def random_binary_states(length):
    '''
    Generate some random test data.
    '''
    return np.random.choice([0, 1], length)


def generate_binary_states(length):
    '''
    Generate all possible binary states of a given length.
    '''
    return np.array(np.meshgrid(*[[0, 1]] * length)).T.reshape(-1, length)


def E(state, theta):
    '''
    Calculate the energy of a state.
    '''
    return np.dot(state.T, np.matmul(theta, state))


def P(state, theta, all_states):
    '''
    Calculate the probability of being in a particular state. Z is the partition function.
    '''

    Z = sum(np.exp(E(s, theta)) for s in all_states)  # partition function
    return np.exp(E(state, theta)) / Z


def P_data(data):
    '''
    Takes given data in `data`, and generates the probability of each state correspondingly.
    '''
    k = data.shape[1]
    unique, counts = np.unique(data, axis=0, return_counts=True)
    state_list = generate_binary_states(k)
    data_prob = np.zeros(len(state_list))
    for i in range(len(state_list)):
        match_indices = np.where(np.all(unique == state_list[i], axis=1))[0]
        if match_indices.size > 0:
            data_prob[i] = counts[match_indices[0]] / len(data)
    return data_prob


# note: does not include realJ like in isingfit.py.
def KL(theta, data, data_probs_list):
    '''
    Returns the KL divergence of two probability distributions. In this case, the KL divergence between model and real probability distributions is calculated.
    '''
    k = data.shape[1]
    theta = theta.reshape(k, k)

    # Generates all possible binary states for k
    state_list = generate_binary_states(k)

    # Initialize the probability arrays
    data_prob, model_prob = data_probs_list, np.zeros(len(state_list))

    # Iterate over each state in state_list
    for j in range(len(state_list)):
        # Always calculate the model probability
        model_prob[j] = P(state_list[j], theta, state_list)

    return stats.entropy(data_prob, model_prob)


def likelihood(J_matrix, net_array):
    '''
    Generates likelihood ratio of estimator being active to inactive.
    '''
    J_xx, J_xs, J_sx = J_matrix[0, 0], J_matrix[0, 1:], J_matrix[1:, 0]
    return np.exp((J_xx+J_xs.dot(net_array)+net_array.T.dot(J_sx)))


def maxent(stimulus: np.ndarray, network: np.ndarray, system_size: float) -> np.ndarray:
    '''
    Given stimulus and network neurons, fits a Maximum Entropy model to determine the probability distribution of the system.  

    Args
    ----
    stimulus: np.ndarray
        A (1,n) array corresponding to the time series of the stimulus neuron over n samples.
    network : np.ndarray
        A (`system_size`-1, n) array corresponding to the time series of the network neurons over n samples.
    system_size : float
        The total number of neurons going into the fitting process.

    ## Returns:
    op_params : np.ndarray
        The optimized parameters corresponding to the J matrix in the MaxEnt model. Sanity check: it should be symmetric.

    **NOTE: stim, net assumed in a form such that columns are time bins, and therefore are transposed further down. It's unideal, but a part of the legacy code for some other files.**
    '''
    print(
        f'BEGINNING MAXENT PROCESS FOR SYSTEM SIZE OF {system_size} NEURONS, STANDBY...\n' + '=' * 65)
    s = time.time()

    # initialize fitting parameters, note that theta should be symmetric
    upper_indices = np.triu_indices(system_size)
    theta0 = np.zeros((system_size, system_size))
    theta0[upper_indices] = np.random.normal(0, 0.3, len(upper_indices[0]))
    theta0 = theta0 + theta0.T

    stim = np.atleast_2d(stimulus).T
    timeshifted_data = np.hstack((stim, network))  # preps for fitting

    # fitting timeshifted data
    timeshifted_probs = P_data(timeshifted_data)

    # print(timeshifted_probs)
    theta0 = theta0.flatten()  # flatten theta to 1d array

    # minimize KL divergence between model and data, reshape to 2d array
    op_params = minimize(KL, theta0, args=(timeshifted_data, timeshifted_probs), method='L-BFGS-B',
                         options={'maxiter': 200}, tol=1e-7).x.reshape(system_size, system_size)

    e = time.time()

    print(
        f'KL divergence MINIMIZED, and J matrix elements FOUND. Proceeding to next step. COMPILATION TIME: {round(e-s, 2)} seconds.\n')

    return op_params
