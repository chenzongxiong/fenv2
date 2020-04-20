import sys
import time
import argparse

import numpy as np
import trading_data as tdata
import log as logging
import constants
import colors

LOG = logging.getLogger(__name__)

methods = constants.METHODS
weights = constants.WEIGHTS
widths = constants.WIDTHS
NB_PLAYS = constants.NB_PLAYS
points = constants.POINTS
UNITS = constants.UNITS


def operator_generator_with_noise():

    mu = 0
    sigma = 1.5
    method = 'sin'
    points = 1000
    with_noise = False
    individual = True
    input_dim = 1
    state = 0
    nb_plays = 50

    if with_noise is False:
        sigma = 0
        mu = 0

    inputs, outputs, multi_outputs = tdata.DatasetGenerator.systhesis_operator_generator(points=points,
                                                                                         nb_plays=nb_plays,
                                                                                         method=method,
                                                                                         mu=mu,
                                                                                         sigma=sigma,
                                                                                         with_noise=with_noise,
                                                                                         individual=individual)
    fname = constants.DATASET_PATH['operators'].format(method=method,
                                                       state=state,
                                                       mu=mu,
                                                       sigma=sigma,
                                                       nb_plays=nb_plays,
                                                       points=points,
                                                       input_dim=input_dim)
    tdata.DatasetSaver.save_data(inputs, outputs, fname)

    if multi_outputs is not None:
        fname_multi = constants.DATASET_PATH['operators_multi'].format(method=method,
                                                                       state=state,
                                                                       mu=mu,
                                                                       sigma=sigma,
                                                                       nb_plays=nb_plays,
                                                                       points=points,
                                                                       input_dim=input_dim)

        tdata.DatasetSaver.save_data(inputs, multi_outputs.T, fname_multi)


def model_generator_with_noise():
    mu = 0
    sigma = 1.5
    method = 'sin'
    points = 1000
    with_noise = False
    activation = 'tanh'
    input_dim = 1
    state = 0
    nb_plays = 50
    diff_weights = True
    units = 50

    if with_noise is False:
        sigma = 0
        mu = 0

    inputs, outputs = tdata.DatasetGenerator.systhesis_model_generator(points=points,
                                                                       nb_plays=nb_plays,
                                                                       method=method,
                                                                       mu=mu,
                                                                       sigma=sigma,
                                                                       activation=activation,
                                                                       with_noise=with_noise,
                                                                       diff_weights=diff_weights,
                                                                       input_dim=input_dim,
                                                                       units=units)

    fname = constants.DATASET_PATH['models'].format(method=method,
                                                    state=state,
                                                    mu=mu,
                                                    sigma=sigma,
                                                    nb_plays=nb_plays,
                                                    points=points,
                                                    input_dim=input_dim,
                                                    activation=activation,
                                                    units=units)

    tdata.DatasetSaver.save_data(inputs, outputs, fname)


def model_genertor_with_mc():
    mu = 0
    sigma = 0.2
    method = 'sin'
    points = 5000
    with_noise = True
    activation = 'tanh'
    # activation = None
    input_dim = 1
    state = 0
    nb_plays = 50
    diff_weights = True
    units = 50

    if with_noise is False:
        sigma = 0
        mu = 0

    inputs = tdata.DatasetGenerator.systhesis_markov_chain_generator(points, mu, sigma)

    inputs, outputs = tdata.DatasetGenerator.systhesis_model_generator(points=points,
                                                                       nb_plays=nb_plays,
                                                                       method=method,
                                                                       mu=mu,
                                                                       sigma=sigma,
                                                                       activation=activation,
                                                                       with_noise=with_noise,
                                                                       diff_weights=diff_weights,
                                                                       input_dim=input_dim,
                                                                       units=units,
                                                                       inputs=inputs)

    fname = constants.DATASET_PATH['models_diff_weights_mc'].format(method=method,
                                                                    state=state,
                                                                    mu=mu,
                                                                    sigma=sigma,
                                                                    nb_plays=nb_plays,
                                                                    points=points,
                                                                    input_dim=input_dim,
                                                                    activation=activation,
                                                                    units=units)
    tdata.DatasetSaver.save_data(outputs, inputs, fname)


def model_nb_plays_generator_with_noise(points=100,
                                        nb_plays=1,
                                        units=1,
                                        activation='tanh',
                                        mu=0,
                                        sigma=1,
                                        with_noise=False,
                                        diff_weights=False):
    # sigma = 7

    method = 'sin'

    run_test = False

    input_dim = 1
    state = 0

    if with_noise is False:
        mu = 0
        sigma = 0

    if diff_weights is True and run_test is True:
        file_key = 'models_diff_weights_test'
    elif diff_weights is True:
        file_key = 'models_diff_weights'
    else:
        file_key = 'models'

    LOG.debug("generate model data for method {}, units {}, nb_plays {}, mu: {}, sigma: {}, points: {}, activation: {}, input_dim: {}".format(method, units, nb_plays, mu, sigma, points, activation, input_dim))

    inputs = None

    start = time.time()
    individual = True
    if individual is False:
        inputs, outputs = tdata.DatasetGenerator.systhesis_model_generator(inputs=inputs,
                                                                           nb_plays=nb_plays,
                                                                           points=points,
                                                                           units=units,
                                                                           mu=mu,
                                                                           sigma=sigma,
                                                                           input_dim=input_dim,
                                                                           activation=activation,
                                                                           with_noise=with_noise,
                                                                           method=method,
                                                                           diff_weights=diff_weights,
                                                                           individual=False)
    else:
        inputs, outputs, individual_outputs = tdata.DatasetGenerator.systhesis_model_generator(inputs=inputs,
                                                                                               nb_plays=nb_plays,
                                                                                               points=points,
                                                                                               units=units,
                                                                                               mu=mu,
                                                                                               sigma=sigma,
                                                                                               input_dim=input_dim,
                                                                                               activation=activation,
                                                                                               with_noise=with_noise,
                                                                                               method=method,
                                                                                               diff_weights=diff_weights,
                                                                                               individual=True)
        outputs = np.hstack([individual_outputs, outputs.reshape(-1, 1)])

    # import ipdb; ipdb.set_trace()
    end = time.time()
    LOG.debug("time cost: {} s".format(end-start))

    fname = constants.DATASET_PATH[file_key].format(method=method,
                                                    activation=activation,
                                                    state=state,
                                                    mu=mu,
                                                    sigma=sigma,
                                                    units=units,
                                                    nb_plays=nb_plays,
                                                    points=points,
                                                    input_dim=input_dim)

    LOG.debug(colors.cyan("Write  data to file {}".format(fname)))
    tdata.DatasetSaver.save_data(inputs, outputs, fname)


def generate_debug_data():
    fname = 'new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv'
    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    points = 1500
    inputs, outputs = inputs[:points], outputs[:points]
    min_price, max_price = inputs.min(), inputs.max()
    cycles = 100
    import ipdb; ipdb.set_trace()

    points_per_half_cycle = 1000
    min_price = -0.4;
    eps = (max_price - min_price) / cycles

    price_list = []

    for i in range(cycles):
        j = (i // 5 + 1)
        if i == 0:
            a = np.linspace(0, min_price, points_per_half_cycle // j)
        else:
            a = np.linspace(max_price-(i-1)*eps, min_price, points_per_half_cycle //j)

        b = np.linspace(min_price, max_price-i*eps, points_per_half_cycle //j)

        price_list.append(np.hstack([a, b]))

    prices = np.hstack(price_list)
    points = prices.shape[0]
    print("points: {}".format(points))
    # import matplotlib.pyplot as plt
    # plt.plot(range(points), prices, '.')
    # plt.show()
    fname = 'new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000-debug-2.csv'
    noises = np.zeros(points)
    tdata.DatasetSaver.save_data(prices, noises, fname)


if __name__ == "__main__":
    # generate_debug_data()

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", dest="operator",
                        required=False,
                        action="store_true",
                        help="generate operators' dataset")
    parser.add_argument("--play", dest="play",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")
    parser.add_argument("--model", dest="model",
                        required=False,
                        action="store_true",
                        help="generate models' dataset")

    parser.add_argument("--nb_plays", dest="nb_plays",
                        required=False,
                        type=int)
    parser.add_argument("--units", dest="units",
                        required=False,
                        type=int)

    parser.add_argument("--GF", dest="GF",
                        required=False,
                        action="store_true",
                        help="generate G & F models' dataset")

    parser.add_argument("--operator-noise", dest="operator_noise",
                        required=False,
                        action="store_true",
                        help="generate operators' dataset")

    parser.add_argument("--play-noise", dest="play_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")
    parser.add_argument('--activation', dest='activation',
                        required=False,
                        default=None,
                        help='acitvation of non-linear layer')
    parser.add_argument("--mc", dest="mc",
                        required=False,
                        action="store_true")
    parser.add_argument("--mu", dest="mu",
                        required=False,
                        type=float)

    parser.add_argument("--sigma", dest="sigma",
                        required=False,
                        type=float)
    parser.add_argument("--points", dest="points",
                        required=False,
                        type=int)

    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")

    parser.add_argument("--model-noise", dest="model_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")

    parser.add_argument("--with-noise", dest="with_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")

    argv = parser.parse_args(sys.argv[1:])

    if argv.operator_noise:
        operator_generator_with_noise()
    if argv.mc:
        model_genertor_with_mc()
    if argv.model_noise:
        # model_generator_with_noise()
        # model_noise_test_generator()
        model_nb_plays_generator_with_noise(argv.points,
                                            argv.nb_plays,
                                            argv.units,
                                            argv.activation,
                                            int(argv.mu),
                                            int(argv.sigma),
                                            with_noise=argv.with_noise,
                                            diff_weights=argv.diff_weights)
