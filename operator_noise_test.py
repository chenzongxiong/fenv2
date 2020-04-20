import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from core import Play
import utils
import trading_data as tdata
import log as logging
import constants


writer = utils.get_tf_summary_writer("./log/operators")
sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS


def fit(inputs, outputs, width, method, true_weight, loss='mse', mu=0, sigma=0.001, loss_file_name="./tmp/operator_loss_history.csv"):
    LOG.debug("timestap is: {}".format(inputs.shape[0]))
    total_timesteps = inputs.shape[0]
    train_timesteps = int(total_timesteps * 0.5)

    batch_size = 1
    EPOCHS = 10000
    epochs = EPOCHS // batch_size
    # epochs = 1
    steps_per_epoch = batch_size
    units = 10

    train_inputs, train_outputs = inputs[:train_timesteps], outputs[:train_timesteps]
    test_inputs, test_outputs = inputs[train_timesteps:], outputs[train_timesteps:]

    play = Play(batch_size=batch_size,
                units=units,
                activation=None,
                network_type=constants.NetworkType.OPERATOR,
                loss=loss)

    start = time.time()
    if loss == 'mse':
        play.fit(train_inputs, train_outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)

        train_loss, metrics = play.evaluate(train_inputs, train_outputs, steps_per_epoch=steps_per_epoch)
        test_loss, metrics = play.evaluate(test_inputs, test_outputs, steps_per_epoch=steps_per_epoch)

        train_predictions = play.predict(train_inputs, steps_per_epoch=1)
        test_predictions = play.predict(test_inputs, steps_per_epoch=1)

        train_mu = train_sigma = test_mu = test_sigma = -1

    elif loss == 'mle':
        play.fit2(train_inputs, mu, sigma, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name)
        train_predictions, train_mu, train_sigma = play.predict2(train_inputs, steps_per_epoch=1)
        test_predictions, test_mu, test_sigma = play.predict2(test_inputs, steps_per_epoch=1)
        train_loss = ((train_outputs - train_predictions) ** 2).mean()
        test_loss = ((test_outputs - test_predictions) ** 2).mean()
        train_loss = float(train_loss)
        test_loss = float(test_loss)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("number of layer is: {}".format(play.number_of_layers))
    LOG.debug("weight: {}".format(play.weight))

    train_predictions = train_predictions.reshape(-1)
    test_predictions = test_predictions.reshape(-1)

    predictions = np.hstack([train_predictions, test_predictions])
    loss = [train_loss, test_loss, train_mu, test_mu, train_sigma, test_sigma]

    return predictions, loss


if __name__ == '__main__':
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    loss_name = 'mse'
    # train dataset
    mu = 0
    sigma = 0.1
    points = 5000
    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["operators_noise"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points)
                inputs, outputs = tdata.DatasetLoader.load_data(fname)
                loss_file_name = constants.FNAME_FORMAT["operators_noise_loss_histroy"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points, loss=loss_name)
                # inputs, outputs = inputs[:40], outputs[:40]
                predictions, loss = fit(inputs, outputs, width, method, weight, loss_name, mu, sigma, loss_file_name)

                fname = constants.FNAME_FORMAT["operators_noise_loss"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points, loss=loss_name)
                tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                fname = constants.FNAME_FORMAT["operators_noise_predictions"].format(method=method, weight=weight, width=width, mu=mu, sigma=sigma, points=points, loss=loss_name)
                tdata.DatasetSaver.save_data(inputs, predictions, fname)
