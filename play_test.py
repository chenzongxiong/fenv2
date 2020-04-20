import sys
import argparse
import time
import numpy as np
import tensorflow as tf

import utils
from core import Play
import log as logging
import constants
import trading_data as tdata

constants.LOG_DIR = "./log/plays"
writer = utils.get_tf_summary_writer("./log/plays")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS
points = constants.POINTS


def fit(inputs, outputs, units, activation, width, true_weight, loss='mse'):

    units = units
    batch_size = 10
    epochs = EPOCHS // batch_size

    steps_per_epoch = batch_size

    total_timesteps = inputs.shape[0]
    train_timesteps = int(total_timesteps * 0.5)

    train_inputs, train_outputs = inputs[:train_timesteps], outputs[:train_timesteps]
    test_inputs, test_outputs = inputs[train_timesteps:], outputs[train_timesteps:]

    import time
    start = time.time()
    play = Play(batch_size=batch_size,
                units=units,
                activation="tanh",
                network_type=constants.NetworkType.PLAY,
                loss=loss,
                debug=False)

    if loss == 'mse':
        play.fit(train_inputs, train_outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)

        train_loss, metrics = play.evaluate(train_inputs, train_outputs, steps_per_epoch=steps_per_epoch)
        test_loss, metrics = play.evaluate(test_inputs, test_outputs, steps_per_epoch=steps_per_epoch)

        train_predictions = play.predict(train_inputs, steps_per_epoch=1)
        test_predictions = play.predict(test_inputs, steps_per_epoch=1)

        train_mu = train_sigma = test_mu = test_sigma = -1
    elif loss == 'mle':
        mu = 0
        sigma = 0.0001
        play.fit2(train_inputs, mu, sigma, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
        train_loss = test_loss = -1
        train_predictions, train_mu, train_sigma = play.predict2(train_inputs, steps_per_epoch=1)
        test_predictions, test_mu, test_sigma = play.predict2(test_inputs, steps_per_epoch=1)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("number of layer is: {}".format(play.number_of_layers))
    LOG.debug("weight: {}".format(play.weight))

    train_predictions = train_predictions.reshape(-1)
    test_predictions = test_predictions.reshape(-1)

    predictions = np.hstack([train_predictions, test_predictions])
    if np.any(np.isnan(predictions)):
        predictions = np.zeros(predictions.shape)
        train_mu = train_sigma = test_mu = test_sigma = -1

    loss = [train_loss, test_loss, train_mu, test_mu, train_sigma, test_sigma]

    return predictions, loss


if __name__ == "__main__":
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", dest="loss",
                        required=True)
    parser.add_argument("--units", dest="units",
                        required=False, type=int)

    argv = parser.parse_args(sys.argv[1:])

    loss_name = argv.loss
    units = argv.units or 1

    activation = "tanh"

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["plays"].format(method=method, weight=weight, width=width, points=points)
                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                # increase *units* in order to increase the capacity of the model
                # for units in _units:
                if True:
                    predictions, loss = fit(inputs, outputs_, units, activation, width, weight, loss_name)
                    fname = constants.FNAME_FORMAT["plays_loss"].format(method=method, weight=weight,
                                                                        width=width, activation=activation, units=units, points=points, loss=loss_name)
                    tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                    fname = constants.FNAME_FORMAT["plays_predictions"].format(method=method, weight=weight,
                                                                               width=width, activation=activation, units=units,
                                                                               points=points, loss=loss_name)
                    tdata.DatasetSaver.save_data(inputs, predictions, fname)
