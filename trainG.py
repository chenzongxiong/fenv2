import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from core import MyModel
import utils
import trading_data as tdata
import log as logging
import constants
import colors

sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS


def fit(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=10, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv"):

    epochs = EPOCHS // batch_size
    epochs = 300
    steps_per_epoch = batch_size

    start = time.time()
    agent = MyModel(batch_size=batch_size,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays)

    agent.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    # agent.weights
    # predictions = agent(inputs)

    predictions = agent.predict(inputs)
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss

if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", dest="loss",
                        required=True)
    parser.add_argument("--units", dest="units",
                        required=False, type=int)



    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS
    batch_size = 10
    points = constants.POINTS
    loss_name = 'mse'
    nb_plays = 4
    nb_plays_ = 4
    # train dataset
    mu = 1.0
    sigma = 0.01
    activation = 'tanh'

    # units = argv.units
    units = 20

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}".format(method, weight, width))
                fname = constants.FNAME_FORMAT["F_predictions"].format(method=method, weight=weight, width=width, points=points, activation='tanh', units=units, sigma=sigma, mu=mu, loss='mse')

                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                # inputs, outputs_ = inputs[:1000], outputs_[:1000]
                inputs, outputs_ = outputs_[:1000], inputs[:1000]
                # inputs, outputs_ = inputs[:40], outputs_[:40]
                # increase *units* in order to increase the capacity of the model
                # for units in _units:
                if True:
                    loss_history_file = constants.FNAME_FORMAT["G_loss_history"].format(method=method, weight=weight,
                                                                                        width=width, activation=activation, units=units,
                                                                                        nb_plays=nb_plays, mu=mu, sigma=sigma, nb_plays_=nb_plays_, batch_size=batch_size, loss=loss_name, points=points)

                    predictions, loss = fit(inputs, outputs_, units, activation, width, weight, method, nb_plays_, batch_size, loss_name, loss_history_file)
                    fname = constants.FNAME_FORMAT["G_loss"].format(method=method, weight=weight,
                                                                    width=width, activation=activation, units=units,
                                                                    nb_plays=nb_plays, nb_plays_=nb_plays_, batch_size=batch_size,
                                                                    loss=loss_name, points=points, mu=mu, sigma=sigma)
                    tdata.DatasetSaver.save_loss({"loss": loss}, fname)
                    fname = constants.FNAME_FORMAT["G_predictions"].format(method=method, weight=weight,
                                                                           width=width, activation=activation, units=units,
                                                                           nb_plays=nb_plays, nb_plays_=nb_plays_, batch_size=batch_size,
                                                                           points=points, loss='mse', mu=mu, sigma=sigma)
                    tdata.DatasetSaver.save_data(inputs, predictions, fname)
