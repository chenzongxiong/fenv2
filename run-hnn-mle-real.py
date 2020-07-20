import os
import sys
import argparse
import math

import time
import numpy as np

import log as logging
from core import MyModel, confusion_matrix
import trading_data as tdata
import constants
import colors
import utils
import tensorflow as tf

LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS


def fit(inputs,
        outputs,
        mu,
        sigma,
        units=1,
        activation='tanh',
        nb_plays=1,
        learning_rate=0.001,
        loss_file_name="./tmp/my_model_loss_history.csv",
        weights_name='model.h5',
        loss_name='mse',
        batch_size=10,
        ensemble=1,
        force_train=False,
        learnable_mu=False):

    epochs = 10000
    # epochs = 6000
    # epochs = 2

    start = time.time()
    input_dim = batch_size

    timestep = 1
    input_dim = inputs.shape[0]
    # timestep = inputs.shape[0] // input_dim
    # steps_per_epoch = inputs.shape[0] // input_dim
    steps_per_epoch = 1

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      learning_rate=learning_rate,
                      ensemble=ensemble,
                      diff_weights=True,
                      learnable_mu=learnable_mu)

    LOG.debug("Learning rate is {}".format(learning_rate))

    preload_weights = False

    if force_train or \
      not os.path.isfile("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays)):
        mymodel.fit2(inputs=inputs,
                     mu=mu,
                     sigma=sigma,
                     outputs=outputs,
                     epochs=epochs,
                     verbose=1,
                     steps_per_epoch=steps_per_epoch,
                     loss_file_name=loss_file_name,
                     preload_weights=preload_weights,
                     weights_fname=weights_fname)
        end = time.time()
        LOG.debug("time cost: {}s".format(end-start))
        LOG.debug("saving weights info")
        mymodel.save_weights(weights_fname)
        LOG.debug("finished saving weights")
    else:
        LOG.debug("already trained, ignore. If you still want to re-train , you can pass flag `force_train`")


def predict(inputs,
            outputs,
            units=1,
            activation='tanh',
            nb_plays=1,
            weights_name='model.h5'):
    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()

    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    start = time.time()

    input_dim = shape[2]
    timestep = shape[1]

    # num_samples = inputs.shape[0] // (input_dim * timestep)
    if input_dim * timestep > inputs.shape[0]:
        # we need to append extra value to make test_inputs and train_outputs have the same
        # keep test_outputs unchanged
        inputs = np.hstack([inputs, np.zeros(input_dim*timestep-test_inputs.shape[0])])

    start = time.time()
    parallel_prediction = True
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=parallel_prediction)

    mymodel.load_weights(weights_fname)
    predictions = mymodel.predict_parallel(inputs)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    predictions = predictions[:outputs.shape[0]]
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return inputs[:outputs.shape[0]], predictions


def trend(prices,
          B,
          mu,
          sigma,
          units=1,
          activation='tanh',
          nb_plays=1,
          weights_name='model.h5',
          trends_list_fname=None,
          ensemble=1):

    # best_epoch = None
    # try:
    #     with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
    #         line = f.read()
    # except FileNotFoundError:
    # # if True:
    #     epochs = []
    #     base = '/'.join(weights_fname.split('/')[:-1])
    #     for _dir in os.listdir(base):
    #         if os.path.isdir('{}/{}'.format(base, _dir)):
    #             try:
    #                 epochs.append(int(_dir.split('-')[-1]))
    #             except ValueError:
    #                 pass

    #     if not epochs:
    #         raise Exception("no trained parameters found")

    #     best_epoch = max(epochs)
    #     best_epoch = 15000
    #     LOG.debug("Best epoch is {}".format(best_epoch))
    #     dirname = '{}-epochs-{}/{}plays'.format(weights_fname[:-3], best_epoch, nb_plays)
    #     if not os.path.isdir(dirname):
    #         # sanity checking
    #         raise Exception("Bugs inside *save_wegihts* or *fit2*")
    #     with open("{}/input_shape.txt".format(dirname), 'r') as f:
    #         line = f.read()


    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()

    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]

    timestep = 1
    shape[1] = timestep

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=True,
                      ensemble=ensemble)

    mymodel.load_weights(weights_fname, extra={'shape': shape})
    guess_trend = mymodel.trend(prices=prices, B=B, mu=mu, sigma=sigma)

    loss = float(-1.0)
    return guess_trend, loss


def plot_graphs_together(price_list, noise_list, mu, sigma,
                         units=1,
                         activation='tanh',
                         nb_plays=1,
                         weights_name='model.h5',
                         trends_list_fname=None, ensemble=1):
    best_epoch = None

    # try:
    #     with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
    #         line = f.read()
    # except FileNotFoundError:
    if True:
        epochs = []
        base = '/'.join(weights_fname.split('/')[:-1])
        for _dir in os.listdir(base):
            if os.path.isdir('{}/{}'.format(base, _dir)):
                try:
                    epochs.append(int(_dir.split('-')[-1]))
                except ValueError:
                    pass

        if not epochs:
            raise Exception("no trained parameters found")

        best_epoch = max(epochs)
        best_epoch = 15000
        LOG.debug("Best epoch is {}".format(best_epoch))
        dirname = '{}-epochs-{}/{}plays'.format(weights_fname[:-3], best_epoch, nb_plays)
        if not os.path.isdir(dirname):
            # sanity checking
            raise Exception("Bugs inside *save_wegihts* or *fit2*")
        with open("{}/input_shape.txt".format(dirname), 'r') as f:
            line = f.read()

    shape = list(map(int, line.split(":")))
    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]

    timestep = 1
    shape[1] = timestep
    parallelism = True
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=parallelism,
                      ensemble=ensemble)

    mymodel.load_weights(weights_fname, extra={'shape': shape, 'parallelism': parallelism, 'best_epoch': best_epoch, 'use_epochs': True})
    mymodel.plot_graphs_together(prices=price_list, noises=noise_list, mu=mu, sigma=sigma)


def visualize(inputs,
              mu=0,
              sigma=1,
              units=1,
              activation='tanh',
              nb_plays=1,
              weights_name='model.h5'):

    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()
    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, "shape must be 3 dimensions"
    input_dim = shape[2]
    # timestep = inputs.shape[0] // input_dim
    timestep = 1
    shape[1] = timestep
    start = time.time()

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)

    mymodel.load_weights(weights_fname, extra={'shape': shape})
    mymodel.visualize_activated_plays(inputs=inputs)


def plot(a, b, trend_list):
    from matplotlib import pyplot as plt
    x = range(1, a.shape[0]+1)
    diff1 = ((a[1:] - a[:-1]) >= 0).tolist()
    diff2 = ((b[1:] - a[:-1]) >= 0).tolist()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(x, a, color='blue')
    ax1.plot(x, b, color='black')

    for index, d1, d2 in zip(x[1:], diff1, diff2):
        if d1 is True and d2 is True:
            ax1.scatter([index], [b[index-1]], marker='^', color='green')
        elif d1 is False and d2 is False:
            ax1.scatter([index], [b[index-1]], marker='^', color='green')
        elif d1 is False and d2 is True:
            ax1.scatter([index], [b[index-1]], marker='s', color='black')
        elif d1 is True and d2 is False:
            ax1.scatter([index], [b[index-1]], marker='s', color='black')

    ax2.plot(x, a, color='blue')
    min_trend_list = trend_list.min(axis=1)
    max_trend_list = trend_list.max(axis=1)
    ax2.fill_between(x, min_trend_list, max_trend_list, facecolor='gray', alpha=0.5, interpolate=True)
    ax3.plot(x, a, color='blue')
    trend_list_ = [trend for trend in  trend_list]
    ax3.boxplot(trend_list_)

    plt.show()
    # fname = "/Users/baymax_testios/Desktop/1.png"
    fname = "./1.png"
    fig.savefig(fname, dpi=400)


def ttest_rel(method1, method2):
    # outputs = np.array(outputs).reshape(-1)
    # guess_prices = np.array(guess_prices).reshape(-1)

    # loss1 =  ((guess_prices - prices[start_pos:end_pos]) ** 2)
    # loss2 = np.abs(guess_prices - prices[start_pos:end_pos])
    # loss3 = (prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1]) ** 2
    # loss4 = np.abs(prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1])

    # LOG.debug("root sum square loss1: {}".format((loss1.sum()/(end_pos-start_pos))**(0.5)))
    # LOG.debug("root sum square loss2: {}".format((loss3.sum()/(end_pos-start_pos))**(0.5)))
    # LOG.debug("total abs loss1: {}".format((loss2.sum()/(end_pos-start_pos))))
    # LOG.debug("total abs loss2: {}".format((loss4.sum()/(end_pos-start_pos))))

    # guess_prices_list = np.array(guess_prices_list)
    pass


def rmse_bucket(ground_truth_noise, ground_truth_price, predict_price, price_steps=10, noise_steps=10):
    diff_price_list = np.abs(ground_truth_price[1:] - ground_truth_price[:-1])
    diff_noise_list = np.abs(ground_truth_noise[1:] - ground_truth_noise[:-1])

    delta_price_step = (np.max(diff_price_list) - np.min(diff_price_list)) / price_steps
    delta_noise_step = (np.max(diff_noise_list) - np.min(diff_noise_list)) / noise_steps

    delta_price_list = [delta_price_step * i + np.min(diff_price_list) for i in range(price_steps+1)]
    delta_noise_list = [delta_noise_step * i + np.min(diff_noise_list) for i in range(noise_steps+1)]

    diff_ground_truth_predict_of_price_list = ground_truth_price[1:] - predict_price

    bucket = { (p, n) : [] for p in range(price_steps+1) for n in range(noise_steps+1)}

    max_val = -1
    i = 0
    for dp, dn, diff in zip(diff_price_list, diff_noise_list, diff_ground_truth_predict_of_price_list):
        _p_idx = (dp - np.min(diff_price_list)) / delta_price_step
        _n_idx = (dn - np.min(diff_noise_list)) / delta_noise_step

        p_idx = math.floor(_p_idx)
        n_idx = math.ceil(_n_idx)

        # import ipdb; ipdb.set_trace()

        p_idx1 = p_idx
        n_idx1 = n_idx

        val = diff * diff
        # val = abs(diff)
        if val > max_val:
            max_val = val

        # single constraints >= p, <= n
        # for _n_idx in range(n_idx, noise_steps+1):
        #     for _p_idx in range(0, p_idx+1):
        #         bucket[(_p_idx, _n_idx)].append((val, dp, dn, i, predict_price[i], ground_truth_price[i+1], ground_truth_price[i]))

        bucket[(p_idx, n_idx)].append((val, dp, dn, i, predict_price[i], ground_truth_price[i+1], ground_truth_price[i]))
        i += 1

    return bucket, delta_price_step, delta_noise_step, delta_price_list, delta_noise_list, max_val


def rmse3d():
    # https://stackoverflow.com/questions/23670178/matplotlib-3d-bar-chart-axis-issue
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np
    from matplotlib import colors as mcolors
    from matplotlib import cm

    base_file = "./new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv"

    trend_file = "./new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-20/nb_plays-20/points-1000/input_dim-1/predictions-mu-0-sigma-110-points-1000/activation#-elu/state#-0/units#-100/nb_plays#-100/ensemble-11/loss-mle/trends-batch_size-1500.csv"
    _, ground_truth_noise = tdata.DatasetLoader.load_data(base_file)
    ground_truth_price, predict_price = tdata.DatasetLoader.load_data(trend_file)

    ground_truth_price = ground_truth_price[:100]
    predict_price = predict_price[:100]
    # ground_truth_noise =
    baseline_price = ground_truth_price[:-1]
    # ground_truth_price = ground_truth_price[1:]
    predict_price = predict_price[1:]
    # ground_truth_noise = ground_truth_noise[1001:1010]
    ground_truth_noise = ground_truth_noise[1000:1100]

    price_steps = 5
    noise_steps = 5

    assert price_steps == noise_steps, "price_steps and noise_steps must be the same"
    _baseline_bucket, delta_price_step, delta_noise_step, delta_price_list, delta_noise_list, baseline_max_rmse = rmse_bucket(ground_truth_noise, ground_truth_price, baseline_price, price_steps, noise_steps)
    _predict_bucket, _, _, _, _, predict_max_rmse  = rmse_bucket(ground_truth_noise, ground_truth_price, predict_price, price_steps, noise_steps)

    max_rmse = predict_max_rmse if predict_max_rmse > baseline_max_rmse else baseline_max_rmse
    rmse_steps = 5
    delta_rmse = max_rmse / rmse_steps
    delta_rmse_list = [i*delta_rmse for i in range(rmse_steps+1)]

    def _helper(ax, _bucket, x, y, zlabel='rmse', color='cyan', func=lambda v: v):
        _zz = np.zeros((price_steps+1, noise_steps+1), dtype=np.float32)

        for k, v in _bucket.items():
            if len(v) != 0:
                _p, _n = k
                # ipdb;ipdb.set_trace()
                # _p * delta_price_step + np.min(diff_price_list)
                _v = [vv[0] for vv in v]
                _pv = [vv[1] for vv in v]
                _nv = [vv[2] for vv in v]
                _zz[k] = func(_v)
                print("{}, ({}, {}), {}, {}, {}, {}".format(k, delta_price_list[_p], delta_noise_list[_n], v, _zz[k],  min(_v), max(_nv)))

        z = _zz.T.ravel()
        print("zz: ", _zz)
        # import ipdb; ipdb.set_trace()

        bottom = np.zeros_like(z)
        ax.bar3d(x, y, bottom, width, depth, z, shade=True, color=color)

        ax.w_xaxis.set_ticks(_x)
        xticks = np.array(["{:.3f}".format(p) for p in delta_price_list])
        ax.w_xaxis.set_ticklabels(xticks)

        ax.w_yaxis.set_ticks(_y + 0.5)
        yticks = np.array(["{:.1f}".format(n) for n in delta_noise_list])
        ax.w_yaxis.set_ticklabels(yticks)

        ax.set_xlabel('$\Delta p$')
        ax.set_ylabel('$\Delta n$')
        ax.set_zlabel(zlabel)
        return z

    fig = plt.figure(constrained_layout=True)

    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    ax1 = fig.add_subplot(spec[0, 0], projection='3d')
    ax2 = fig.add_subplot(spec[0, 1], projection='3d', sharez=ax1)

    ax3 = fig.add_subplot(spec[1, 0], projection='3d')
    ax4 = fig.add_subplot(spec[1, 1], projection='3d', sharez=ax3)

    _x = np.arange(len(delta_price_list))
    _y = np.arange(len(delta_noise_list))

    _xx, _yy = np.meshgrid(_x, _y)
    print("_xx: ", _xx)
    print("_yy: ", _yy)
    # import ipdb; ipdb.set_trace()
    x, y = _xx.ravel(), _yy.ravel()

    width = 0.5
    depth = 0.5

    print("width: ", width, ", depth: ", depth)

    values = np.linspace(0.2, 1., x.shape[0])
    colors = cm.rainbow(values)

    if baseline_max_rmse > predict_max_rmse:
        _helper(ax2, _predict_bucket, x, y, zlabel='PREDICT-RMSE',
                # color=mcolors.CSS4_COLORS['darkorange'],
                color=colors,
                func=lambda v: (sum(v)/len(v))**0.5)
                # func=lambda v: (sum(v)/len(v)))
        _helper(ax1, _baseline_bucket, x, y, zlabel='BASELINE-RMSE',
                # color=mcolors.CSS4_COLORS['dodgerblue'],
                color=colors,
                func=lambda v: (sum(v)/len(v))**0.5)
                # func=lambda v: (sum(v)/len(v)))
    else:
        _helper(ax1, _baseline_bucket, x, y, zlabel='BASELINE-RMSE',
                # color=mcolors.CSS4_COLORS['dodgerblue'],
                color=colors,
                func=lambda v: (sum(v)/len(v))**0.5)
                # func=lambda v: (sum(v)/len(v)))
        _helper(ax2, _predict_bucket, x, y, zlabel='PREDICT-RMSE',
                # color=mcolors.CSS4_COLORS['darkorange'],
                color=colors,
                func=lambda v: (sum(v)/len(v))**0.5)
                # func=lambda v: (sum(v)/len(v)))

    print("================================================================================")
    _helper(ax3, _baseline_bucket, x, y, zlabel='BASELINE-COUNTS',
            # color=mcolors.CSS4_COLORS['dodgerblue'],
            color=colors,
            func=lambda v: len(v))
    _helper(ax4, _predict_bucket, x, y, zlabel='PREDICT-COUNTS',
            # color=mcolors.CSS4_COLORS['darkorange'],
            color=colors,
            func=lambda v: len(v))

    plt.show()
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # print("Plot RMSE 3D")
    # rmse3d()
    # import ipdb; ipdb.set_trace()

    LOG.debug(colors.red("Test multiple plays"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", dest="batch_size",
                        default=1000,
                        type=int)
    parser.add_argument("--nb_plays", dest="nb_plays",
                        default=-1,
                        type=int)
    parser.add_argument("--units", dest="units",
                        default=-1,
                        type=int)
    parser.add_argument("--activation", dest="activation",
                        default=None,
                        type=str)
    parser.add_argument("--__nb_plays__", dest="__nb_plays__",
                        default=2,
                        type=int)
    parser.add_argument("--__units__", dest="__units__",
                        default=5,
                        type=int)
    parser.add_argument("--__activation__", dest="__activation__",
                        default=None,
                        type=str)
    parser.add_argument('--trend', dest='trend',
                        default=False, action='store_true')
    parser.add_argument('--predict', dest='predict',
                        default=False, action='store_true')
    parser.add_argument('--plot', dest='plot',
                        default=False, action='store_true')
    parser.add_argument('--visualize_activated_plays', dest='visualize_activated_plays',
                        default=False, action='store_true')
    parser.add_argument('--mu', dest='mu',
                        default=0,
                        type=float)
    parser.add_argument('--sigma', dest='sigma',
                        default=110,
                        type=float)

    parser.add_argument('--__mu__', dest='__mu__',
                        default=0,
                        type=float)
    parser.add_argument('--__sigma__', dest='__sigma__',
                        default=110,
                        type=float)
    parser.add_argument('--ensemble', dest='ensemble',
                        default=2,  # start from 1
                        type=int)
    parser.add_argument('--force-train', dest='force_train',
                        default=False, action='store_true')

    parser.add_argument('--learnable-mu', dest='learnable_mu',
                        default=False, action='store_true')
    parser.add_argument('--method', dest='method',
                        default='sin', type=str)

    argv = parser.parse_args(sys.argv[1:])
    # Hyper Parameters
    # learning_rate = 0.003
    learning_rate = 0.07

    batch_size = argv.batch_size

    # loss_name = 'mse'
    loss_name = 'mle'

    method = argv.method

    # method = 'mixed'
    # method = 'noise'
    interp = 1
    # do_prediction = False
    do_prediction = argv.predict
    do_confusion_matrix = False
    mc_mode = False

    do_trend = argv.trend
    do_plot = argv.plot
    do_visualize_activated_plays = argv.visualize_activated_plays
    ensemble = argv.ensemble
    with_noise = True

    diff_weights = True

    run_test = False

    # mu = 0
    # sigma = 110
    mu = argv.mu
    sigma = argv.sigma
    if sigma == int(sigma):
        sigma = int(sigma)
    if mu == int(mu):
        mu = int(mu)

    points = 0
    input_dim = 1
    ############################## ground truth #############################
    nb_plays = argv.nb_plays
    # units is 10000 special for dataset comes from simulation
    units = argv.units
    state = 0
    # activation = 'tanh'
    # activation = None
    activation = argv.activation
    ############################## predicitons #############################
    __nb_plays__ = argv.__nb_plays__
    __units__ = argv.__units__
    # __nb_plays__ = 2
    # __units__ = 2

    __state__ = 0
    __activation__ = argv.__activation__
    # __activation__ = 'relu'
    # __activation__ = None
    # __activation__ = 'tanh'
    # __mu__ = 2.60

    __mu__ = argv.__mu__
    __sigma__ = argv.__sigma__

    if method == 'noise':
        with_noise = True

    if with_noise is False:
        mu = 0
        sigma = 0

    if diff_weights is True:
        # input_file_key = 'models_diff_weights'
        # loss_file_key = 'models_diff_weights_loss_history'
        if mc_mode is True:
            weights_file_key = 'models_diff_weights_mc_saved_weights'
        else:
            weights_file_key = 'models_diff_weights_saved_weights'
        # predictions_file_key = 'models_diff_weights_predictions'
        weights_file_key = 'models_diff_weights_mc_saved_weights'
    else:
        # input_file_key = 'models'
        # loss_file_key = 'models_loss_history'
        # weights_file_key = 'models_saved_weights'
        # predictions_file_key = 'models_predictions'
        raise

    # weights_file_key = 'models_diff_weights_mc_stock_model_saved_weights'
    weights_file_key = 'models_diff_weights_saved_weights'
    # XXXX: place weights_fname before run_test
    weights_fname = constants.DATASET_PATH[weights_file_key].format(method=method,
                                                                    activation=activation,
                                                                    state=state,
                                                                    mu=mu,
                                                                    sigma=sigma,
                                                                    units=units,
                                                                    nb_plays=nb_plays,
                                                                    points=points,
                                                                    input_dim=input_dim,
                                                                    __activation__=__activation__,
                                                                    __state__=__state__,
                                                                    __units__=__units__,
                                                                    __nb_plays__=__nb_plays__,
                                                                    loss=loss_name,
                                                                    ensemble=ensemble,
                                                                    batch_size=batch_size)
    if interp != 1:
        if do_prediction is False:
            raise
        if run_test is True:
            raise
        elif run_test is False:
            raise
    elif interp == 1:
        if run_test is True:
            raise
        elif run_test is False:
            if diff_weights is True:
                input_file_key = 'models_diff_weights'
                loss_file_key = 'models_diff_weights_loss_history'
                predictions_file_key = 'models_diff_weights_predictions'
            else:
                raise

    # if do_trend is True:
    ################### markov chain #############################
    if mc_mode is True:
        input_file_key = 'models_diff_weights_mc_stock_model'
        loss_file_key = 'models_diff_weights_mc_stock_model_loss_history'
        predictions_file_key = 'models_diff_weights_mc_stock_model_predictions'
        if do_trend is True:
            predictions_file_key = 'models_diff_weights_mc_stock_model_trends'
            trends_list_file_key = 'models_diff_weights_mc_stock_model_trends_list'
    else:
        # input_file_key = 'models_diff_weights_mc'
        # loss_file_key = 'models_diff_weights_mc_loss_history'
        # predictions_file_key = 'models_diff_weights_mc_predictions'

        input_file_key = 'models_diff_weights'
        loss_file_key = 'models_diff_weights_loss_history'
        predictions_file_key = 'models_diff_weights_predictions'


    fname = constants.DATASET_PATH[input_file_key].format(interp=interp,
                                                          method=method,
                                                          activation=activation,
                                                          state=state,
                                                          mu=mu,
                                                          sigma=sigma,
                                                          units=units,
                                                          nb_plays=nb_plays,
                                                          points=points,
                                                          input_dim=input_dim)

    LOG.debug("Load data from file: {}".format(colors.cyan(fname)))
    if do_prediction is True and do_trend is True:
        raise Exception("both do predictions and do_trend are True")

    # Debug Dima hysteresis behaviours
    # fname = 'new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000-debug-5.csv'
    inputs, outputs = tdata.DatasetLoader.load_data(fname, columns=['inputs'])
    if do_trend is False:
        # inputs, outputs = inputs[:points], outputs[:points]
        pass
    if mc_mode is True:
        # inputs, outputs = outputs, inputs
        pass
    else:
        # inputs, outputs = outputs, inputs
        # gap = 5
        # inputs, outputs = inputs[::gap], outputs[::gap]
        # # inputs = np.arange(800)[::4].astype(np.float32)
        # # inputs = np.zeros(800)[::4].astype(np.float32)
        # # mu = 0
        # # sigma = 0.5
        # # points = 200
        # # noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
        # # inputs += noise
        # mu1 = 4
        # sigma1 = 2.5
        # inputs = tdata.DatasetGenerator.systhesis_markov_chain_generator(200, mu1, sigma1)

        pass

    # inputs, outputs = outputs, inputs
    # inputs, outputs = inputs[:2000], outputs[:2000]
    # It's for debug variables
    # inputs, outputs = inputs[:1500*20], outputs[:1500*20]
    # inputs, outputs = inputs[::20], outputs[::20]

    loss_history_file = constants.DATASET_PATH[loss_file_key].format(interp=interp,
                                                                     method=method,
                                                                     activation=activation,
                                                                     state=state,
                                                                     mu=mu,
                                                                     sigma=sigma,
                                                                     units=units,
                                                                     nb_plays=nb_plays,
                                                                     points=points,
                                                                     input_dim=input_dim,
                                                                     __activation__=__activation__,
                                                                     __state__=__state__,
                                                                     __units__=__units__,
                                                                     __nb_plays__=__nb_plays__,
                                                                     loss=loss_name,
                                                                     ensemble=ensemble,
                                                                     batch_size=batch_size)

    predicted_fname = constants.DATASET_PATH[predictions_file_key].format(interp=interp,
                                                                          method=method,
                                                                          activation=activation,
                                                                          state=state,
                                                                          mu=mu,
                                                                          sigma=sigma,
                                                                          units=units,
                                                                          nb_plays=nb_plays,
                                                                          points=points,
                                                                          input_dim=input_dim,
                                                                          __activation__=__activation__,
                                                                          __state__=__state__,
                                                                          __units__=__units__,
                                                                          __nb_plays__=__nb_plays__,
                                                                          loss=loss_name,
                                                                          ensemble=ensemble,
                                                                          batch_size=batch_size)

    if mc_mode is True and do_trend is True:
        trends_list_fname = constants.DATASET_PATH[trends_list_file_key].format(interp=interp,
                                                                                method=method,
                                                                                activation=activation,
                                                                                state=state,
                                                                                mu=mu,
                                                                                sigma=sigma,
                                                                                units=units,
                                                                                nb_plays=nb_plays,
                                                                                points=points,
                                                                                input_dim=input_dim,
                                                                                __activation__=__activation__,
                                                                                __state__=__state__,
                                                                                __units__=__units__,
                                                                                __nb_plays__=__nb_plays__,
                                                                                loss=loss_name,
                                                                                ensemble=ensemble,
                                                                                batch_size=batch_size)



    LOG.debug('############################  SETTINGS #########################################')
    LOG.debug('# Learning Rate: {}'.format(learning_rate))
    LOG.debug('# points: {}'.format(points))
    LOG.debug('# nb_plays: {}'.format(nb_plays))
    LOG.debug('# units: {}'.format(units))
    LOG.debug('# activation: {}'.format(activation))
    LOG.debug("# mu: {}".format(mu))
    LOG.debug("# sigma: {}".format(sigma))
    LOG.debug("# state: {}".format(state))
    LOG.debug('# __nb_plays__: {}'.format(__nb_plays__))
    LOG.debug('# __units__: {}'.format(__units__))
    LOG.debug('# __activation__: {}'.format(__activation__))
    LOG.debug("# __mu__: {}".format(__mu__))
    LOG.debug("# __sigma__: {}".format(__sigma__))
    LOG.debug("# __state__: {}".format(__state__))

    LOG.debug("# do_prediction: {}".format(do_prediction))
    LOG.debug("# do_trend: {}".format(do_trend))
    LOG.debug("# do_fit: {}".format(not (do_prediction and do_trend)))
    LOG.debug("# mc_mode: {}".format(mc_mode))

    LOG.debug('# train_fname: {}'.format(colors.cyan(fname)))
    LOG.debug('# predicted_fname: {}'.format(colors.cyan(predicted_fname)))
    LOG.debug('# weights_fname: {}'.format(colors.cyan(weights_fname)))

    LOG.debug('################################################################################')

    # input(colors.red("Press Enter to continue..."))

    # try:
    #     predicted_fname = 'new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-20/nb_plays-20/points-1000/input_dim-1/predictions-mu-0-sigma-110-points-1000/activation#-elu/state#-0/units#-100/nb_plays#-100/ensemble/loss-mle/trends-batch_size-1500.csv'
    #     a, b = tdata.DatasetLoader.load_data(predicted_fname)
    #     # inp, trend_list = tdata.DatasetLoader.load_data(trends_list_fname)
    #     # assert np.allclose(a, inp, atol=1e-5)
    #     confusion = confusion_matrix(a, b)
    #     LOG.debug(colors.purple("confusion matrix is: {}".format(confusion)))

    #     # plot(a, b, trend_list)

    #     hnn_rsme = (((b[:-1] - a[:-1]) ** 2).mean())**(0.5)
    #     baseline_rsme =  (((a[1:] - a[:-1]) ** 2).mean())**(0.5)
    #     # loss2 = np.abs(guess_prices - prices[start_pos:end_pos])
    #     # loss3 = (prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1]) ** 2
    #     # loss4 = np.abs(prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1])
    #     LOG.debug("hnn-RMSE: {}".format(hnn_rsme))
    #     LOG.debug("baseline-RMSE: {}".format(baseline_rsme))

    #     sys.exit(0)
    # except FileNotFoundError:
    #     LOG.warning("Not found prediction file, no way to create confusion matrix")

    if mc_mode is True and do_trend is True:
        predictions, loss = trend(prices=inputs[:batch_size*2],
                                  B=outputs[:batch_size*2],
                                  mu=__mu__,
                                  sigma=__sigma__,
                                  units=__units__,
                                  activation=__activation__,
                                  nb_plays=__nb_plays__,
                                  weights_name=weights_fname,
                                  trends_list_fname=trends_list_fname,
                                  ensemble=ensemble)
        # inputs = inputs[batch_size:batch_size+predictions.shape[-1]]
        # inputs = inputs[batch_size:batch_size+predictions.shape[-1]]
        # inputs = inputs[1000:1100]
        import ipdb; ipdb.set_trace()
        inputs = inputs[1510:1610]
        # inputs = inputs[1510:1515]
    elif do_visualize_activated_plays is True:
        LOG.debug(colors.red("Load weights from {}, DO VISUALIZE ACTIVATED PLAYS".format(weights_fname)))
        visualize(inputs=inputs[:batch_size],
                  mu=__mu__,
                  sigma=__sigma__,
                  units=__units__,
                  activation=__activation__,
                  nb_plays=__nb_plays__,
                  weights_name=weights_fname)
        sys.exit(0)
    elif do_prediction is True:

        LOG.debug(colors.red("Load weights from {}".format(weights_fname)))
        # import ipdb; ipdb.set_trace()
        # inputs, outputs = inputs[:batch_size], outputs[:batch_size]
        predictions, best_epoch = predict(inputs=inputs,
                                           outputs=outputs,
                                           units=__units__,
                                           activation=__activation__,
                                           nb_plays=__nb_plays__,
                                           weights_name=weights_fname)
        if best_epoch is not None:
            predicted_fname = "{}-epochs-{}.csv".format(predicted_fname[:-4], best_epoch)

    elif do_plot is True:
        inputs, outputs = inputs[:batch_size], outputs[:batch_size]
        plot_graphs_together(price_list=inputs, noise_list=outputs, mu=__mu__, sigma=__sigma__,
                             weights_name=weights_fname,
                             units=__units__,
                             activation=__activation__,
                             nb_plays=__nb_plays__,
                             ensemble=ensemble)
        sys.exit(0)
    else:
        LOG.debug("START to FIT via {}".format(colors.red(loss_name.upper())))
        # _inputs, _outputs = inputs[:1700], outputs[:1700]
        # train_inputs, train_outputs = _inputs[:1500], _outputs[:1500]
        # test_inputs, test_outputs = _inputs[1500:], _outputs[1500:]

        # _inputs, _outputs = inputs[:1700], outputs[:1700]
        # train_inputs, train_outputs = _inputs[:1300], _outputs[:1300]
        # test_inputs, test_outputs = _inputs[1300:], _outputs[1300:]

        _inputs = inputs[:1700]
        train_inputs, test_inputs = _inputs[:1300], _inputs[1300:]

        fit(inputs=train_inputs,
            # outputs=train_outputs,
            outputs=None,
            mu=__mu__,
            sigma=__sigma__,
            units=__units__,
            activation=__activation__,
            nb_plays=__nb_plays__,
            learning_rate=learning_rate,
            loss_file_name=loss_history_file,
            weights_name=weights_fname,
            loss_name=loss_name,
            batch_size=batch_size,
            ensemble=ensemble,
            force_train=argv.force_train,
            learnable_mu=argv.learnable_mu)

        inputs, predictions = predict(inputs=test_inputs,
                                      # outputs=test_outputs,
                                      outputs=None,
                                      units=__units__,
                                      activation=__activation__,
                                      nb_plays=__nb_plays__,
                                      weights_name=weights_fname)

    LOG.debug("Write data into predicted_fname: {}".format(predicted_fname))
    tdata.DatasetSaver.save_data(inputs, predictions, predicted_fname)
    LOG.debug('========================================FINISHED========================================')
