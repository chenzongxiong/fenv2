import sys

sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..")

import constants
import numpy as np
import trading_data as tdata

if __name__ == '__main__':
    method = 'stock'
    activation = None
    state = 0
    mu = 0
    sigma = 20
    units = 0
    nb_plays = 0
    points = 0
    input_dim = 1

    input_file_key = 'models_diff_weights'
    fname = constants.DATASET_PATH[input_file_key].format(method=method,
                                                          activation=activation,
                                                          state=state,
                                                          mu=mu,
                                                          sigma=sigma,
                                                          units=units,
                                                          nb_plays=nb_plays,
                                                          points=points,
                                                          input_dim=input_dim)
    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    diff = outputs[1:] - outputs[:-1]
    _mu = diff.mean()
    _std = diff.std()
    rescale_diff = (diff-_mu)/_std


    rescale_outputs = [0]
    N = outputs.shape[0] - 1
    for i in range(N):
        rescale_outputs.append(rescale_outputs[-1] + rescale_diff[i])

    rescale_outputs = np.array(rescale_outputs)

    fname = constants.DATASET_PATH[input_file_key].format(method='rescaled-stock',
                                                          activation=activation,
                                                          state=state,
                                                          mu=mu,
                                                          sigma=sigma,
                                                          units=units,
                                                          nb_plays=nb_plays,
                                                          points=points,
                                                          input_dim=input_dim)
    tdata.DatasetSaver.save_data(inputs, rescale_outputs, fname)
