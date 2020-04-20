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


def fit(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=1, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv", learning_rate=0.001, weights_fname='model.h5', mu=0, sigma=1.0):

    # epochs = EPOCHS // batch_size
    epochs = 10000
    steps_per_epoch = batch_size

    start = time.time()
    input_dim = 5000
    # timestep = 100 // input_dim
    # import ipdb; ipdb.set_trace()
    timestep = inputs.shape[0] // input_dim
    agent = MyModel(input_dim=input_dim,
                    timestep=timestep,
                    units=units,
                    activation="elu",
                    nb_plays=nb_plays,
                    network_type=constants.NetworkType.PLAY)

    # agent.load_weights(weights_fname)
    LOG.debug("Learning rate is {}".format(learning_rate))
    if loss == 'mse':
        agent.fit(inputs, outputs, verbose=1, epochs=epochs,
                steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name, learning_rate=learning_rate)
    elif loss == 'mle':
        agent.fit2(inputs, mu, sigma, outputs, verbose=1, epochs=epochs,
                   steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name, learning_rate=learning_rate)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    agent.weights

    agent.save_weights(weights_fname)
    predictions = agent.predict(inputs)

    predictions = agent.predict(inputs)
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss



if __name__ == "__main__":
    # LOG.debug(colors.red("Test multiple plays"))
    # inputs, outputs = tdata.DatasetLoader.load_data(constants.DATASET_PATH['models'].format(method='sin',
    #                                                                                         state=0,
    #                                                                                         mu=0,
    #                                                                                         sigma=0,
    #                                                                                         nb_plays=50,
    #                                                                                         points=1000,
    #                                                                                         input_dim=1,
    #                                                                                         units=50,
    #                                                                                         activation='tanh'))

    # predictions, loss = fit(inputs, outputs, units=100, activation='elu', nb_plays=50, learning_rate=0.05,
    #                         weights_fname=constants.DATASET_PATH['models_saved_weights'].format(
    #                             method='sin',
    #                             state=0,
    #                             mu=0,
    #                             sigma=0,
    #                             units=50,
    #                             activation='tanh',
    #                             nb_plays=50,
    #                             __units__=50,
    #                             __activation__='elu',
    #                             __nb_plays__=50,
    #                             __state__=0,
    #                             points=1000,
    #                             input_dim=1,
    #                             loss='mse',
    #                             ))

    # prediction_fname = constants.DATASET_PATH['models_predictions'].format(method='sin',
    #                                                                        state=0,
    #                                                                        mu=0,
    #                                                                        sigma=0,
    #                                                                        nb_plays=50,
    #                                                                        units=50,
    #                                                                        activation='tanh',
    #                                                                        __nb_plays__=50,
    #                                                                        __units__=50,
    #                                                                        __activation__='elu',
    #                                                                        __state__=0,
    #                                                                        points=1000,
    #                                                                        input_dim=1,
    #                                                                        loss='mse')
    # tdata.DatasetSaver.save_data(inputs, predictions, prediction_fname)

    LOG.debug(colors.red("Test multiple mc with MLE"))
    inputs, outputs = tdata.DatasetLoader.load_data(constants.DATASET_PATH['models_diff_weights_mc'].format(method='sin',
                                                                                                            state=0,
                                                                                                            mu=0,
                                                                                                            sigma=0.2,
                                                                                                            nb_plays=50,
                                                                                                            points=5000,
                                                                                                            input_dim=1,
                                                                                                            units=50,
                                                                                                            activation='tanh'))

    predictions, loss = fit(inputs, outputs,
                            units=100,
                            activation='elu',
                            nb_plays=100,
                            learning_rate=0.01,
                            mu=0,
                            sigma=0.2,
                            loss='mle',
                            weights_fname=constants.DATASET_PATH['models_diff_weights_mc_saved_weights'].format(
                                method='sin',
                                state=0,
                                mu=0,
                                sigma=0.2,
                                units=50,
                                activation='tanh',
                                nb_plays=50,
                                __units__=100,
                                __activation__='elu',
                                __nb_plays__=100,
                                __state__=0,
                                points=5000,
                                input_dim=1,
                                loss='mle'
                                ))

    prediction_fname = constants.DATASET_PATH['models_diff_weights_mc_predictions'].format(method='sin',
                                                                                           state=0,
                                                                                           mu=0,
                                                                                           sigma=0.2,
                                                                                           nb_plays=50,
                                                                                           units=50,
                                                                                           activation='tanh',
                                                                                           __nb_plays__=100,
                                                                                           __units__=100,
                                                                                           __activation__='elu',
                                                                                           __state__=0,
                                                                                           points=5000,
                                                                                           input_dim=1,
                                                                                           loss='mle')
    tdata.DatasetSaver.save_data(inputs, predictions, prediction_fname)
