import sys
sys.path.append('.')
sys.path.append('..')

import argparse

import numpy as np
import trading_data as tdata
import log as logging
import colors

LOG = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--activation', dest='activation',
                        required=False,
                        default=None,
                        help='acitvation of non-linear layer')
    parser.add_argument("--mu", dest="mu",
                        required=False,
                        type=float)
    parser.add_argument("--sigma", dest="sigma",
                        required=False,
                        type=float)
    parser.add_argument("--lr", dest="lr",
                        required=False, default=0.001,
                        type=float)
    parser.add_argument("--points", dest="points",
                        required=False,
                        type=int)
    parser.add_argument("--nb_plays", dest="nb_plays",
                        required=False,
                        type=int)
    parser.add_argument("--units", dest="units",
                        required=False,
                        type=int)
    parser.add_argument("--__units__", dest="__units__",
                        required=False,
                        type=int)
    parser.add_argument("--__activation__", dest="__activation__",
                        required=False,
                        type=str)
    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")
    parser.add_argument('--markov-chain', dest='mc',
                        required=False,
                        action="store_true")

    argv = parser.parse_args(sys.argv[1:])

    lr = argv.lr
    mu = int(argv.mu)
    sigma = int(argv.sigma)
    points = argv.points
    markov_chain = argv.mc

    activation = argv.activation
    nb_plays = argv.nb_plays
    units = argv.units
    __units__ = argv.__units__
    __activation__ = argv.__activation__

    if markov_chain is True:
        # log/lstm-mle-lr-0.005-activation-None-nb_plays-20-units-1-points-1000-mu-0-sigma-110-__activation__-tanh-__units__-256.log
        fname="./log/lstm-mle-lr-{lr}-activation-{activation}-nb_plays-{nb_plays}-units-{units}-points-{points}-mu-{mu}-sigma-{sigma}-__activation__-{__activation__}-__units__-{__units__}.log".format(
            activation=activation,
            lr=lr,
            mu=mu,
            sigma=sigma,
            nb_plays=nb_plays,
            units=1,
            __units__=__units__,
            __activation__=__activation__,
            points=points
        )
        # ./new-dataset/lstm/diff_weights/method-sin/activation-None/state-0/input_dim-1/mu-0/sigma-110/units-1/nb_plays-20/points-2000/markov_chain/units\#-1/activation\#-tanh/loss-mle/

        log_fname="./new-dataset/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/markov_chain/units#-{__units__}/activation#-{__activation__}/loss-{loss}/mle-loss-lr-{lr}.csv".format(
            method='sin',
            activation=activation,
            state=0,
            input_dim=1,
            mu=mu,
            sigma=sigma,
            units=1,
            nb_plays=nb_plays,
            points=2000,
            __units__=__units__,
            __activation__=__activation__,
            lr=lr,
            loss='mle'
            )
    elif argv.diff_weights:
        fname="./log/lstm-diff-weights-activation-{activation}-lr-{lr}-mu-{mu}-sigma-{sigma}-nb_play-{nb_plays}-units-{units}-__units__-{__units__}-points-{points}.log".format(
            activation=activation,
            lr=lr,
            mu=mu,
            sigma=sigma,
            nb_plays=nb_plays,
            units=units,
            __units__=__units__,
            points=points
        )

        log_fname="./new-dataset/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/mse-loss-lr-{lr}.csv".format(
            method='sin',
            activation=activation,
            state=0,
            input_dim=1,
            mu=mu,
            sigma=sigma,
            units=units,
            nb_plays=nb_plays,
            points=points,
            __units__=__units__,
            lr=lr
            )
    else:
        fname="./log/lstm-activation-{activation}-lr-{lr}-mu-{mu}-sigma-{sigma}-nb_play-{nb_plays}-units-{units}-__units__-{__units__}-points-{points}.log".format(
            activation=activation,
            lr=lr,
            mu=mu,
            sigma=sigma,
            nb_plays=nb_plays,
            units=units,
            __units__=__units__,
            points=points
        )

        log_fname="./new-dataset/lstm/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/mse-loss-lr-{lr}.csv".format(
            method='sin',
            activation=activation,
            state=0,
            input_dim=1,
            mu=mu,
            sigma=sigma,
            units=units,
            nb_plays=nb_plays,
            points=points,
            __units__=__units__,
            lr=lr
            )

    LOG.debug(colors.cyan("extract loss from fname: {}".format(fname)))
    LOG.debug(colors.cyan("to loss history: {}".format(log_fname)))

    loss_history = []
    fp = open(fname, 'r')

    split_ratio = 0.6
    validation_ratio = 0.05
    if markov_chain:
        interesting_part = 'mean_squared_error'
    else:
        interesting_part = "{}/{}".format(int(split_ratio*points*(1-validation_ratio)),
                                          int(split_ratio*points*(1-validation_ratio)))

    for line in fp:
        if interesting_part in line:
            seg = line.split()
            try:
                loss_history.append(float(seg[7]))
            except IndexError:
                print("ERROR: lines: {}".format(line))

    loss_history = np.array(loss_history)
    tdata.DatasetSaver.save_data(np.arange(loss_history.shape[0], dtype=np.int32), loss_history, log_fname)
    fp.close()
