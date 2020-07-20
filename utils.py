import os
import threading
import h5py

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation

import log as logging
import colors

LOG = logging.getLogger(__name__)

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())


def update(i, *fargs):
    inputs = fargs[0]
    outputs = fargs[1]
    ax = fargs[2]
    colors = fargs[3]
    mode = fargs[4]
    step = fargs[5]
    if mode == "snake":
        xlim = fargs[6]
        ylim = fargs[7]
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if i % 100 == 0:
        LOG.info("Update animation frame: {}, step: {}".format(i, step))

    s = [n*8 for n in range(step)]
    if mode == "sequence":
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+step, x], outputs[i:i+step, x], color=colors[x])
    elif mode == "snake":
        inputs_len = inputs.shape[0]
        for x in range(len(colors)):
            # ax.scatter(inputs[0:inputs_len, x], outputs[0:inputs_len, x], color='cyan')
            ax.scatter(inputs[:, x], outputs[:, x], color='cyan')
        for x in range(len(colors)):
            ax.scatter(inputs[i:i+step, x], outputs[i:i+step, x], color=colors[x], s=s)
            # ax.scatter(inputs[i+inputs_len:i+step+inputs_len, x], outputs[i+inputs_len:i+step+inputs_len, x], color=colors[x], s=s)


def save_animation(inputs, outputs, fname, xlim=None, ylim=None,
                   colors=["black"], step=1, mode="sequence"):
    assert inputs.shape == outputs.shape
    assert mode in ["sequence", "snake"], "mode must be 'sequence' or 'snake'."
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    if xlim is None:
        # xlim = [np.min(inputs) - 1, np.max(inputs) + 1]
        xlim = [np.min(inputs) - 0.1, np.max(inputs) + 0.1]
    if ylim is None:
        ylim = [np.min(outputs) - 1, np.max(outputs) + 1]

    if len(inputs.shape) == 1:
        inputs = inputs.reshape(-1, 1)
        outputs = outputs.reshape(-1, 1)
    if not isinstance(colors, list):
        colors = [colors]

    assert len(colors) == outputs.shape[1]

    fig, ax = plt.subplots(figsize=(30, 15))
    fig.set_tight_layout(True)
    points = inputs.shape[0]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fargs=(inputs, outputs, ax, colors, mode, step, xlim, ylim)

    anim = None
    if mode == "sequence":
        anim = FuncAnimation(fig, update, frames=np.arange(0, points, step),
                             fargs=fargs, interval=400)
    elif mode == "snake":
        frame_step = step // 2
        if frame_step == 0:
            frame_step = 2

        anim = FuncAnimation(fig, update, frames=np.arange(0, points, frame_step),
                             fargs=fargs, interval=400)

    anim.save(fname, dpi=40, writer='imagemagick')


COLORS = ["blue", "black", "orange", "cyan", "red", "magenta", "yellow", "green"]

def generate_colors(length=1):
    if (length >= len(COLORS)):
        LOG.error(colors.red("Doesn't have enough colors"))
        raise
    return COLORS[:length]


class TFSummaryFileWriter(object):
    _writer = None
    _lock = threading.Lock()

    def __new__(cls, fpath="."):
        import tensorflow as tf

        if not cls._writer:
            with cls._lock:
                if not cls._writer:
                    cls._writer = tf.summary.FileWriter(fpath)
        return cls._writer


def get_tf_summary_writer(fpath):
    writer = TFSummaryFileWriter(fpath)
    return writer


_SESSION = None


def get_session(debug=False, interactive=False):
    # https://github.com/tensorflow/tensorflow/issues/5448
    # import multiprocessing as mp
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    import tensorflow as tf
    from tensorflow.python import debug as tf_debug

    global _SESSION
    if _SESSION is not None:
        return _SESSION

    if debug is True:
        _SESSION = tf.keras.backend.set_session(
            tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:1234"))
    elif interactive is True:
        _SESSION = tf.InteractiveSession()
    else:
        _SESSION = tf.keras.backend.get_session()
        # config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False,
        #                         intra_op_parallelism_threads=os.cpu_count())
        # config.gpu_options.allow_growth = True
        # _SESSION = tf.Session(config=config)

    return _SESSION


def clear_session():
    import tensorflow as tf
    tf.keras.backend.clear_session()


def init_tf_variables():
    import tensorflow as tf
    # sess = tf.keras.backend.get_session()
    sess = get_session()
    init = tf.global_variables_initializer()
    sess.run(init)


def read_saved_weights(fname=None):
    f = h5py.File(fname, 'r')

    for k in list(f.keys())[::-1]:
        for kk in list(f[k].keys())[::-1]:
            for kkk in list(f[k][kk].keys())[::-1]:
                print("Layer *{}*, {}: {}".format(colors.red(kk.upper()), colors.red(kkk), list(f[k][kk][kkk])))
    f.close()


def build_play(play, inputs):
    if not play.built:
        play.build(inputs)
    return play


def build_p3(p2, j):
    import tensorflow as tf
    return tf.reduce_sum(tf.cumprod(p2[:, j:], axis=1), axis=1)


def slide_window_average(arr, window_size=5, step=1):
    assert len(arr.shape) == 1, colors.red("slide window only support 1-dim")
    if window_size == 1:
        return arr

    N = arr.shape[0]
    stacked_array = np.vstack([ arr[i: 1 + N + i - window_size:step] for i in range(window_size) ])
    avg = np.concatenate([stacked_array.mean(axis=0), arr[-window_size+1:]])
    return avg


# def _parents(op):
#   return set(input.op for input in op.inputs)

# def parents(op, indent=0, ops_have_been_seen=set()):
#     # for op in ops:
#     #     _ops = list(set(input.op for input in op.inputs))
#     #     print("_ops: {}".format(_ops))]
#     #     parents(_ops, indent=indent+1, ops_have_been_seen=ops_have_been_seen)
#     print(len(op.inputs))
#     for inp in op.inputs:
#         print(op.name, '-->', inp.op.name)
#         parents(inp.op)

#     print("Indent: {}".format(indent))


# def children(op):
#   return set(op for out in op.outputs for op in out.consumers())

# def get_graph():
#   """Creates dictionary {node: {child1, child2, ..},..} for current
#   TensorFlow graph. Result is compatible with networkx/toposort"""
#   import tensorflow as tf
#   ops = tf.get_default_graph().get_operations()
#   return {op: children(op) for op in ops}



# def print_tf_graph(graph):
#   """Prints tensorflow graph in dictionary form."""
#   for node in graph:
#     for child in graph[node]:
#       print("%s -> %s" % (node.name, child.name))
#     print("--------------------------------------------------------------------------------")


# import networkx as nx

# def plot_graph(G):
#     '''Plot a DAG using NetworkX'''
#     def mapping(node):
#         return node.name
#     G = nx.DiGraph(G)
#     nx.relabel_nodes(G, mapping, copy=False)
#     nx.draw(G, cmap = plt.get_cmap('jet'), with_labels=True)
#     plt.show()


def plot_internal_transaction(hysteresis_info, i=None, predicted_price=None, **kwargs):
    fname1 = '../simulation/training-dataset/mu-0-sigma-20.0-points-1700/{}-brief.csv'.format(i+1300)
    fname2 = '../simulation/training-dataset/mu-0-sigma-20.0-points-1700/{}-true-detail.csv'.format(i+1300)
    fname3 = '../simulation/training-dataset/mu-0-sigma-20.0-points-1700/{}-fake-detail.csv'.format(i+1300)
    fname4 = './new-dataset/lstm/lstm5/price_vs_price/units-1300/capacity-128/ensemble-2000/predictions.csv'
    # import ipdb; ipdb.set_trace()
    # _data = np.loadtxt(fname1, delimiter=',')
    gt_data = np.loadtxt(fname2, delimiter=',')
    gt_data = gt_data[-1, 0]
    fake_data = np.loadtxt(fname3, delimiter=',')
    lstm_data = np.loadtxt(fname4, delimiter=',')
    lstm_data = lstm_data[i-1, 1]

    mu = kwargs.pop('mu', 0)
    sigma = kwargs.pop('sigma', 1)
    ensemble = kwargs.pop('ensemble', 0)
    # fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
    # fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=False)
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col')
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    # fig, axes = plt.subplots(2, 2, sharex='col')
    # ax1, ax2, ax3, ax4 = axes[0, 0], axes[1, 0], axes[0, 1], axes[1, 1]
    plot_simulation_info(i-1, ax1)
    plot_hysteresis_info(hysteresis_info, i, predicted_price=predicted_price, ax=ax2)
    # plt.show()
    guess_price_seq = kwargs.pop('guess_price_seq', None)
    bk_list = kwargs.pop('bk_list', None)
    if guess_price_seq is not None:
        # fig1, (ax3, ax4) = plt.subplots(2, sharex=True, figsize=(20, 20))
        fig1, (ax3, ax4, ax5) = plt.subplots(3, sharex=False, figsize=(10, 10))
        guess_price_seq = guess_price_seq.reshape(-1)
        bk_list = bk_list.reshape(-1)
        plot_price_span(guess_price_seq, gt_data, lstm_data, ax3)
        plot_price_distribution(guess_price_seq, ax4)
        plot_noise_distribution(bk_list, ax5)

        if mu is None and sigma is None:
            fname = './frames/{}-distribution.png'.format(i)
        else:
            fname = './frames-mu-{}-sigma-{}-ensemble-{}/{}-distribution.png'.format(mu, sigma, ensemble, i)

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig1.savefig(fname, dpi=100)

    if mu is None and sigma is None:
        fname = './frames/{}.png'.format(i)
    else:
        fname = './frames-mu-{}-sigma-{}-ensemble-{}/{}.png'.format(mu, sigma, ensemble, i)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=100)


def plot_simulation_info(i, ax=None):
    if ax is None:
        fig, (ax, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))

    fname1 = '../simulation/training-dataset/mu-0-sigma-20.0-points-1700/{}-brief.csv'.format(1300+i)
    fname2 = '../simulation/training-dataset/mu-0-sigma-20.0-points-1700/{}-true-detail.csv'.format(1300+i)
    fname3 = '../simulation/training-dataset/mu-0-sigma-20.0-points-1700/{}-fake-detail.csv'.format(1300+i)

    _data = np.loadtxt(fname1, delimiter=',')
    true_data = np.loadtxt(fname2, delimiter=',')
    fake_data = np.loadtxt(fname3, delimiter=',')

    fake_B1, fake_B2, fake_B3, _B1, _B2, _B3 = _data[0], _data[1], _data[2], _data[3], _data[4], _data[5]
    fake_price_list, fake_stock_list = fake_data[:, 0], fake_data[:, 1]
    price_list, stock_list = true_data[:, 0], true_data[:, 1]
    fake_l = 10 if len(fake_price_list) == 1 else len(fake_price_list)
    l = 10 if len(price_list) == 1 else len(price_list)
    fake_B1, fake_B2, fake_B3 = np.array([fake_B1]*fake_l), np.array([fake_B2]*fake_l), np.array([fake_B3]*fake_l)
    _B1, _B2, _B3 = np.array([_B1]*l), np.array([_B2]*l), np.array([_B3]*l)

    fake_l = 10 if len(fake_price_list) == 1 else len(fake_price_list)
    l = 10 if len(price_list) == 1 else len(price_list)

    fake_B1, fake_B2, fake_B3 = np.array([fake_B1]*fake_l), np.array([fake_B2]*fake_l), np.array([fake_B3]*fake_l)
    _B1, _B2, _B3 = np.array([_B1]*l), np.array([_B2]*l), np.array([_B3]*l)

    ax.plot(fake_price_list, fake_B1, 'r', fake_price_list, fake_B2, 'c--', fake_price_list, fake_B3, 'k--')
    ax.plot(price_list, _B1, 'r', price_list, _B2, 'c', price_list, _B3, 'k-')
    ax.plot(fake_price_list, fake_stock_list, color='blue', marker='s', markersize=2, linestyle='--')
    ax.plot(price_list, stock_list, color='blue', marker='.', markersize=4)
    ax.set_xlabel("prices")
    ax.set_ylabel("#Noise")
    # plt.show()


def plot_hysteresis_info(hysteresis_info, i=None, predicted_price=None, **kwargs):
    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.figure()

    colors = [mcolors.CSS4_COLORS['orange']]

    for index, info in enumerate(hysteresis_info):
        guess_hysteresis_list = info[0]
        prices = np.array([g[0] for g in guess_hysteresis_list])
        noise = np.array([g[1] for g in guess_hysteresis_list])
        l = len(prices)
        prev_original_prediction = np.array([info[1]] * l)
        curr_original_prediction = np.array([info[2]] * l)
        bk = np.array([info[3]] * l)
        prev_price = np.array([info[4]] * l)
        curr_price = np.array([info[5]] * l)
        _predicted_price = np.array([predicted_price] * l)
        vertical_line = np.linspace(noise.min(), noise.max(), l)
        if index == len(hysteresis_info) - 1:
            ax.plot(prices, prev_original_prediction, color='red', label='start noise', linewidth=1)
            ax.plot(prices, curr_original_prediction, color='black', label='target noise', linewidth=1)

            ax.plot(_predicted_price, vertical_line, color='orange', label='predicted price', linewidth=1, linestyle='solid')

            ax.plot(prices, noise, color=colors[index % len(colors)], marker='o', markersize=4, label='steps finding root', linewidth=1)
        else:
            ax.plot(prices, prev_original_prediction, color='red', label=None, linewidth=1)
            ax.plot(prices, curr_original_prediction, color='black', label=None, linewidth=1)

            ax.plot(_predicted_price, vertical_line, color='orange', label=None, linewidth=1, linestyle='solid')

            ax.plot(prices, noise, color=colors[index % len(colors)], marker='o', markersize=4, label=None, linewidth=1)

    ax.legend(loc='upper right', shadow=True, fontsize='large')
    ax.set_xlabel("prices")
    ax.set_ylabel("noise")

def plot_price_span(guess_price_seq, gt_data, lstm_data, ax):
    x = range(guess_price_seq.shape[0])
    ax.plot(x, guess_price_seq, marker='.', linewidth=1, label='hnn guess')
    # groud truth
    ax.plot(x, [gt_data]*guess_price_seq.shape[0], marker='+', linewidth=1, color='blue', label='ground truth')
    # HNN
    avg_guess_price = guess_price_seq.mean()
    ax.plot(x, [avg_guess_price]*guess_price_seq.shape[0], marker='x', linewidth=1, color='orange', label='hnn avg')
    # LSTM
    ax.plot(x, [lstm_data]*guess_price_seq.shape[0], marker='*', linewidth=1, color='green', label='lstm')
    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    ax.set_xlabel('')
    ax.set_ylabel('prices')


import scipy.stats as stats

def plot_price_distribution(guess_price_seq, ax):
    ax.hist(guess_price_seq, bins=20)

    mu = guess_price_seq.mean()
    sigma = guess_price_seq.std()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma))

    ax.set_xlabel('price')
    ax.set_ylabel('occurrence')


def plot_noise_distribution(bk_list, ax):
    ax.hist(bk_list, bins=20)
    mu = bk_list.mean()
    sigma = bk_list.std()
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma))

    ax.set_xlabel('noise')
    ax.set_ylabel('occurrence')


_CACHE = None


def get_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = dict()
    return _CACHE


def sentinel_marker():
    return 'SENTINEL'


if __name__ == "__main__":
    import numpy as np
    # arr = np.arange(100)
    # # arr1 = slide_window_average(arr, 1)
    # # arr2 = slide_window_average(arr, 2)
    # arr3 = slide_window_average(arr, 3)
    # plot_internal_transaction(None, 0)
    plot_simulation_info(19)
