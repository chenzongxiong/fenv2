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


def fit(inputs,
        outputs,
        units=1,
        activation='tanh',
        nb_plays=1,
        learning_rate=0.001,
        loss_file_name="./tmp/my_model_loss_history.csv",
        weights_name='model.h5'):

    epochs = 200
    # steps_per_epoch = batch_size

    start = time.time()
    input_dim = 1
    timestep = inputs.shape[0] // input_dim

    steps_per_epoch = 1

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)
    # mymodel.load_weights(weights_fname)
    LOG.debug("Learning rate is {}".format(learning_rate))
    mymodel.fit(inputs,
                outputs,
                verbose=1,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                loss_file_name=loss_file_name,
                learning_rate=learning_rate)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))
    LOG.debug("print weights info")
    # mymodel.weights

    mymodel.save_weights(weights_fname)

    predictions = mymodel.predict(inputs)
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))
    return predictions, loss


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
    predictions_list = []

    input_dim = shape[2]
    timestep = shape[1]
    num_samples = inputs.shape[0] // (input_dim * timestep)

    start = time.time()
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)

    mymodel.load_weights(weights_fname)
    for i in range(num_samples):
        LOG.debug("Predict on #{} sample".format(i+1))
        pred = mymodel.predict(inputs[i*(input_dim*timestep): (i+1)*(input_dim*timestep)])

        predictions_list.append(pred)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))

    predictions = np.hstack(predictions_list)
    outputs = outputs[:predictions.shape[-1]]
    loss = ((predictions - outputs) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))

    return predictions, loss


if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    # Hyper Parameters
    learning_rate = 0.1
    loss_name = 'mse'

    method = 'sin'
    # method = 'mixed'
    # method = 'noise'
    interp = 10

    with_noise = True
    diff_weights = True

    run_test = False
    do_prediction = True

    mu = 0
    sigma = 2

    points = 1000
    input_dim = 1
    ############################## ground truth #############################
    nb_plays = 20
    units = 20
    state = 0
    activation = 'tanh'
    # activation = None
    ############################## predicitons #############################
    __nb_plays__ = 20
    __units__ = 20
    __state__ = 0
    __activation__ = 'tanh'

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
                                                                    loss=loss_name)

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
    import ipdb; ipdb.set_trace()
    inputs, grouth_truth = tdata.DatasetLoader.load_data(fname)


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
                                                                     loss=loss_name)

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
                                                                          loss=loss_name)

    if do_prediction is True:
        LOG.debug(colors.red("Load weights from {}".format(weights_fname)))
        predictions, loss = predict(inputs=inputs,
                                    outputs=grouth_truth,
                                    units=__units__,
                                    activation=__activation__,
                                    nb_plays=__nb_plays__,
                                    weights_name=weights_fname)
        inputs = inputs[:predictions.shape[-1]]
    else:
        predictions, loss = fit(inputs=inputs,
                                outputs=grouth_truth,
                                units=__units__,
                                activation=__activation__,
                                nb_plays=__nb_plays__,
                                learning_rate=learning_rate,
                                loss_file_name=loss_history_file,
                                weights_name=weights_fname)

    tdata.DatasetSaver.save_data(inputs, predictions, predicted_fname)
