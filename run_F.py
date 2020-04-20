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


def fit(inputs,
        outputs,
        units=1,
        activation='tanh',
        nb_plays=1,
        learning_rate=0.001,
        loss_file_name="./tmp/my_model_loss_history.csv",
        weights_name='model.h5'):

    epochs = 6000
    # steps_per_epoch = batch_size

    start = time.time()
    input_dim = 10
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


def generate_Gdata_from_mc(mu,
                           sigma,
                           activation,
                           nb_plays=1,
                           weights_name='model.h5'):
    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()
    shape = list(map(int, line.split(":")))

    assert len(shape) == 3, colors.red("shape must be 3 dimensions")

    predictions_list = []

    input_dim = shape[2]
    timestep = shape[1]

    # num_samples = inputs.shape[0] // (input_dim * timestep)
    num_samples = 1
    points = num_samples * timestep * input_dim
    inputs = tdata.DatasetGenerator.systhesis_markov_chain_generator(points, mu, sigma)

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

    predictions = np.hstack(predictions_list)

    return inputs, predictions


if __name__ == "__main__":
    LOG.debug(colors.red("Test multiple plays"))

    # Hyper Parameters
    learning_rate = 0.01
    loss_name = 'mse'

    method = 'sin'
    # method = 'mixed'
    # method = 'noise'

    generated_Gdata = True

    interp = 1
    do_prediction = False

    with_noise = True
    diff_weights = True

    run_test = False

    use_inversion = True

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
    __units__ = 20
    __state__ = 0
    __activation__ = 'tanh'
    # __activation__ = 'relu'
    # __activation__ = None
    __nb_plays__ = 20
    ############################ For markov chain ##########################
    __mu__ = 0
    __sigma__ = 3.5

    if method == 'noise':
        with_noise = True

    if with_noise is False:
        mu = 0
        sigma = 0

    if use_inversion is False:
        raise Exception(colors.red("F is an inverted neural network, use_inversion must be True"))

    if use_inversion is False:
        if run_test is False:
            if diff_weights is True:
                input_file_key = 'models_diff_weights'
                loss_file_key = 'models_diff_weights_loss_history'
                weights_file_key = 'models_diff_weights_saved_weights'
                predictions_file_key = 'models_diff_weights_predictions'
            else:
                input_file_key = 'models'
                loss_file_key = 'models_loss_history'
                weights_file_key = 'models_saved_weights'
                predictions_file_key = 'models_predictions'
        elif run_test is True:
            raise

    elif use_inversion is True:
        if run_test is False:
            if diff_weights is True:
                invert_file_key = 'models_diff_weights_invert'
                input_file_key = 'models_diff_weights'
                predictions_file_key = 'models_diff_weights_invert_predictions'
                loss_file_key = 'models_diff_weights_invert_loss_history'
                weights_file_key = 'models_diff_weights_invert_saved_weights'
        elif run_test is True:
            raise

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
    # method = 'noise'
    # sigma = 0.5

    if interp != 1:
        if use_inversion is True:
            if run_test is False:
                if diff_weights is True:
                    input_file_key = 'models_diff_weights_invert_interp'
                    predictions_file_key = 'models_diff_weights_invert_interp_predictions'
                else:
                    raise
            elif run_test is True:
                raise
        elif use_inversion is False:
            if run_test is False:
                if diff_weights is True:
                    input_file_key = 'models_diff_weights_interp'
    if generated_Gdata is True:
        if diff_weights is True:
            predictions_file_key = 'models_diff_weights_mc'
        else:
            raise


    if do_prediction is True and generated_Gdata is True:
        raise Exception(colors.red("both do_prediction and generated_Gdata are True"))


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

    saved_invert_fname = constants.DATASET_PATH[invert_file_key].format(interp=interp,
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
    inputs, ground_truth = tdata.DatasetLoader.load_data(fname)

    if use_inversion is True and do_prediction is False:
        if interp == 1:
            LOG.debug(colors.red("swap inputs and outputs..."))
            inputs, ground_truth = ground_truth, inputs
        clip_seq = inputs.shape[0] // 100
        inputs = inputs[:clip_seq*100]
        ground_truth = ground_truth[:clip_seq*100]

        tdata.DatasetSaver.save_data(inputs, ground_truth, saved_invert_fname)

    if do_prediction is False:
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
    if generated_Gdata is True:
        __activation__ = None
        LOG.debug(colors.red("Generated Gdata, Load weights from {}".format(weights_fname)))
        inputs, predictions = generate_Gdata_from_mc(mu=__mu__,
                                                     sigma=__sigma__,
                                                     nb_plays=__nb_plays__,
                                                     activation=__activation__,
                                                     weights_name=weights_fname)


        predicted_fname = constants.DATASET_PATH[predictions_file_key].format(interp=interp,
                                                                              method=method,
                                                                              activation=__activation__,
                                                                              state=state,
                                                                              mu=__mu__,
                                                                              sigma=__sigma__,
                                                                              units=units,
                                                                              nb_plays=nb_plays,
                                                                              points=points,
                                                                              input_dim=input_dim,
                                                                              __activation__=__activation__,
                                                                              __state__=__state__,
                                                                              __units__=__units__,
                                                                              __nb_plays__=__nb_plays__,
                                                                              loss=loss_name)

    elif do_prediction is True:
        LOG.debug(colors.red("Do predictions, Load weights from {}".format(weights_fname)))
        predictions, loss = predict(inputs=inputs,
                                    outputs=ground_truth,
                                    units=__units__,
                                    activation=__activation__,
                                    nb_plays=__nb_plays__,
                                    weights_name=weights_fname)
        inputs = inputs[:predictions.shape[-1]]
    else:
        predictions, loss = fit(inputs=inputs,
                                outputs=ground_truth,
                                units=__units__,
                                activation=__activation__,
                                nb_plays=__nb_plays__,
                                learning_rate=learning_rate,
                                loss_file_name=loss_history_file,
                                weights_name=weights_fname)

    tdata.DatasetSaver.save_data(inputs, predictions, predicted_fname)
