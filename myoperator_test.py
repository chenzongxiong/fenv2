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


def fit(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=1, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv", learning_rate=0.001, weights_fname='model.h5'):

    # epochs = EPOCHS // batch_size
    epochs = 5000
    steps_per_epoch = batch_size

    start = time.time()
    input_dim = 1000
    # timestep = 100 // input_dim
    # import ipdb; ipdb.set_trace()
    timestep = inputs.shape[0] // input_dim
    agent = MyModel(input_dim=input_dim,
                    timestep=timestep,
                    units=units,
                    activation="tanh",
                    nb_plays=nb_plays,
                    network_type=constants.NetworkType.OPERATOR)
    # agent.load_weights(weights_fname)
    LOG.debug("Learning rate is {}".format(learning_rate))
    agent.fit(inputs, outputs, verbose=1, epochs=epochs,
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


def predict(inputs, outputs, units=1, activation='tanh', width=1, weight=1.0, method='sin', nb_plays=1, batch_size=1, loss='mse', loss_file_name="./tmp/my_model_loss_history.csv", learning_rate=0.001, weights_name='model.h5'):

    steps_per_epoch = batch_size

    start = time.time()
    predictions_list = []
    input_dim = 100
    # timestep = 100 // input_dim
    timestep = inputs.shape[0] // input_dim
    timestep = 10
    start = time.time()
    agent = MyModel(batch_size=batch_size,
                    input_dim=input_dim,
                    timestep=timestep,
                    units=units,
                    activation=activation,
                    nb_plays=nb_plays)

    agent.load_weights(weights_fname)
    for i in range(9):
        LOG.debug("Predict on #{} sample".format(i+1))
        predictions = agent.predict(inputs[i*1000: (i+1)*1000])
        predictions_list.append(predictions)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")

    _predictions = np.hstack(predictions_list)
    loss = ((_predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    # import ipdb; ipdb.set_trace()

    return _predictions, loss


if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple operators"))

    inputs, outputs = tdata.DatasetLoader.load_data(constants.DATASET_PATH['operators'].format(method='sin',
                                                                                               state=0,
                                                                                               mu=0,
                                                                                               sigma=0,
                                                                                               nb_plays=20,
                                                                                               points=1000,
                                                                                               input_dim=1))

    predictions, loss = fit(inputs, outputs, units=100, activation='elu', nb_plays=20, learning_rate=0.1,
                            weights_fname=constants.DATASET_PATH['operators_saved_weights'].format(
                                method='sin',
                                state=0,
                                mu=0,
                                sigma=0,
                                nb_plays=20,
                                __nb_plays__=40,
                                points=1000,
                                input_dim=1,
                                loss='mse',
                                ))

    prediction_fname = constants.DATASET_PATH['operators_prediction'].format(method='sin',
                                                                             state=0,
                                                                             mu=0,
                                                                             sigma=0,
                                                                             nb_plays=20,
                                                                             __nb_plays__=40,
                                                                             points=1000,
                                                                             input_dim=1,
                                                                             loss='mse')
    tdata.DatasetSaver.save_data(inputs, predictions, prediction_fname)
