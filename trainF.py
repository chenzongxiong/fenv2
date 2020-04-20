import sys
import argparse
import time
import numpy as np
import tensorflow as tf

import utils
from core import Play, MyModel
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


def fit(inputs, outputs, units, activation, width, true_weight, loss='mse', mu=0, sigma=0.01, loss_file_name="./tmp/trainF-loss.csv", nb_plays=1, learning_rate=0.1, weights_fname="model.h5"):
    # mu = float(mu)
    # sigma = float(sigma)
    # fname = constants.FNAME_FORMAT['mc'].format(mu=mu, sigma=sigma, points=inputs.shape[-1])
    # try:
    #     B, _ = tdata.DatasetLoader.load_data(fname)
    # except:
    B = tdata.DatasetGenerator.systhesis_markov_chain_generator(inputs.shape[-1], mu, sigma)
    # fname = constants.FNAME_FORMAT['mc'].format(points=inputs.shape[-1], mu=mu, sigma=sigma)
    # tdata.DatasetSaver.save_data(B, B, fname)

    units = units
    # batch_size = 1
    input_dim = 10
    timestep = 900
    epochs = 2500
    # epochs = EPOCHS // batch_size
    # steps_per_epoch = batch_size
    steps_per_epoch = 1
    train_inputs, train_outputs = inputs, outputs

    import time
    start = time.time()
    agent = MyModel(# batch_size=batch_size,
                    timestep=timestep,
                    input_dim=input_dim,
                   units=units,
                   activation="tanh",
                   nb_plays=nb_plays)
    agent.load_weights(weights_fname)
    agent.fit(inputs, outputs, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch, loss_file_name=loss_file_name,
              learning_rate=learning_rate)
    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    predictions = agent.predict(inputs)
    agent.save_weights(weights_fname)

    prices = agent.predict(B)
    B = B.reshape(-1)
    prices = prices.reshape(-1)
    return B, prices, predictions


if __name__ == "__main__":
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    _units = constants.UNITS

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", dest="loss",
                        required=False)
    parser.add_argument("--mu", dest="mu",
                        required=False,
                        type=float)
    parser.add_argument("--sigma", dest="sigma",
                        required=False,
                        type=float)
    parser.add_argument("--units", dest="units",
                        required=False,
                        type=int)

    argv = parser.parse_args(sys.argv[1:])

    learning_rate = 0.01
    # loss_name = argv.loss
    loss_name = 'mse'

    mu = 0
    # sigma = 0.01
    sigma = 2
    nb_plays = 20
    nb_plays_ = 20
    units = 20

    points = 1000
    state = 0
    activation = "tanh"

    for method in methods:
        for weight in weights:
            for width in widths:
                LOG.debug("Processing method: {}, weight: {}, width: {}, units: {}, nb_plays: {}, mu: {}, sigma: {}, points: {}, state: {}".format(method, weight, width, units, nb_plays, mu, sigma, points, state))
                # fname = constants.FNAME_FORMAT["models_noise"].format(method=method,
                #                                                       weight=weight,
                #                                                       width=width,
                #                                                       nb_plays=nb_plays,
                #                                                       units=units,
                #                                                       mu=mu,
                #                                                       sigma=sigma,
                #                                                       points=points)


                # fname = constants.FNAME_FORMAT['F_interp'].format(method=method,
                #                                                   weight=weight,
                #                                                   width=width,
                #                                                   nb_plays=nb_plays,
                #                                                   units=units,
                #                                                   points=points,
                #                                                   mu=mu,
                #                                                   sigma=sigma,
                #                                                   nb_plays_=nb_plays_,
                #                                                   batch_size=1,
                #                                                   state=state,
                #                                                   loss=loss_name)

                interp = 10
                fname = constants.FNAME_FORMAT["models_nb_plays_noise_interp"].format(method=method,
                                                                                      weight=weight,
                                                                                      width=width,
                                                                                      nb_plays=nb_plays,
                                                                                      units=units,
                                                                                      points=points,
                                                                                      mu=mu,
                                                                                      sigma=sigma,
                                                                                      interp=interp)

                inputs, outputs_ = tdata.DatasetLoader.load_data(fname)
                # inputs, outputs_ = outputs_, inputs  # F neural network
                inputs, outputs_ = outputs_[:9000], inputs[:9000]
                if True:
                    loss_file_name = constants.FNAME_FORMAT['F_interp_loss_history'].format(method=method,
                                                                                     weight=weight,
                                                                                     width=width,
                                                                                     nb_plays=nb_plays,
                                                                                     units=units,
                                                                                     mu=mu,
                                                                                     sigma=sigma,
                                                                                     points=points,
                                                                                     loss=loss_name,
                                                                                     nb_plays_=nb_plays_,
                                                                                     batch_size=1,
                                                                                     state=state)
                    weights_fname = constants.FNAME_FORMAT['F_interp_saved_weights'].format(method=method,
                                                                                     weight=weight,
                                                                                     width=width,
                                                                                     nb_plays=nb_plays,
                                                                                     units=units,
                                                                                     mu=mu,
                                                                                     sigma=sigma,
                                                                                     points=points,
                                                                                     loss=loss_name,
                                                                                     nb_plays_=nb_plays_,
                                                                                     batch_size=1,
                                                                                     state=state)

                    B, prices, predictions = fit(inputs=inputs,
                                                 outputs=outputs_,
                                                 units=units,
                                                 activation=activation,
                                                 width=width,
                                                 true_weight=weight,
                                                 loss=loss_name,
                                                 mu=mu,
                                                 sigma=sigma,
                                                 loss_file_name=loss_file_name,
                                                 nb_plays=nb_plays_,
                                                 learning_rate=learning_rate,
                                                 weights_fname=weights_fname)

                    fname = constants.FNAME_FORMAT['F_interp'].format(method=method,
                                                                      weight=weight,
                                                                      width=width,
                                                                      nb_plays=nb_plays,
                                                                      units=units,
                                                                      points=points,
                                                                      mu=mu,
                                                                      sigma=sigma,
                                                                      nb_plays_=nb_plays_,
                                                                      batch_size=1,
                                                                      state=state,
                                                                      loss=loss_name)

                    tdata.DatasetSaver.save_data(inputs, outputs_, fname)

                    fname = constants.FNAME_FORMAT['F_interp_predictions'].format(method=method,
                                                                                  weight=weight,
                                                                                  width=width,
                                                                                  nb_plays=nb_plays,
                                                                                  units=units,
                                                                                  mu=mu,
                                                                                  sigma=sigma,
                                                                                  points=points,
                                                                                  loss=loss_name,
                                                                                  nb_plays_=nb_plays_,
                                                                                  batch_size=1,
                                                                                  state=state)

                    tdata.DatasetSaver.save_data(inputs, predictions, fname)

                    # fname = constants.FNAME_FORMAT['F_predictions'].format(method=method,
                    #                                                        weight=weight,
                    #                                                        width=width,
                    #                                                        nb_plays=nb_plays,
                    #                                                        units=units,
                    #                                                        mu=mu,
                    #                                                        sigma=sigma,
                    #                                                        points=points,
                    #                                                        loss=loss_name,
                    #                                                        nb_plays_=nb_plays_,
                    #                                                        batch_size=1,
                    #                                                        state=state)

                    # tdata.DatasetSaver.save_data(B, prices, fname)
