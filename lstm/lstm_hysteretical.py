import sys

sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..")


import os
import argparse
import time
import tensorflow as tf
import numpy as np

import log as logging
import trading_data as tdata
import constants
import colors
import utils

LOG = logging.getLogger(__name__)
session = utils.get_session()

# input vs. output
def lstm(input_fname, units, epochs=1000, weights_fname=None, force_train=False, learning_rate=0.001):

    _train_inputs, _train_outputs = tdata.DatasetLoader.load_train_data(input_fname)
    _test_inputs, _test_outputs = tdata.DatasetLoader.load_test_data(input_fname)

    train_inputs = _train_inputs.reshape(-1, 1, 1)
    train_outputs = _train_outputs.reshape(-1, 1, 1)
    # learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    start = time.time()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(units),
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    if force_train or not os.path.isfile(weights_fname):
        model.fit(train_inputs, train_outputs, epochs=epochs, verbose=1,
                  # callbacks=[early_stopping_callback],
                  validation_split=0.05,
                  shuffle=False)
        os.makedirs(os.path.dirname(weights_fname), exist_ok=True)
        model.save_weights(weights_fname)
    else:
        model.load_weights(weights_fname)

    end = time.time()
    LOG.debug(colors.red("time cost: {}s".format(end- start)))


    test_inputs = _test_inputs.reshape(-1, 1, 1)
    test_outputs = _test_outputs.reshape(-1)

    predictions = model.predict(test_inputs)
    pred_outputs = predictions.reshape(-1)
    rmse = np.sqrt(np.mean((pred_outputs - test_outputs) ** 2))

    LOG.debug(colors.red("LSTM rmse: {}".format(rmse)))

    return _test_inputs, pred_outputs, rmse, end-start


def mle_loss(model, mu, sigma, activation='tanh'):
    # TODO: add elu activation
    def tanh_loss(y_true, y_pred):
        # Extract weights from layers
        lstm_kernel = model.layers[1].cell.kernel
        lstm_recurrent_kernel = model.layers[1].cell.recurrent_kernel
        units = model.layers[1].cell.units
        h_tm1 = model.layers[1].states[0]  # previous memory state
        c_tm1 = model.layers[1].states[1]  # previous carry state

        dense_kernel = model.layers[2].kernel

        inputs = model.layers[0].output[0, :, :]

        z = tf.keras.backend.dot(inputs, lstm_kernel)
        z += tf.keras.backend.dot(h_tm1, lstm_recurrent_kernel)
        if model.layers[1].cell.use_bias:
            z = tf.keras.backend.bias_add(z, model.layers[1].cell.bias)

        w_i = lstm_kernel[:, :units]
        w_f = lstm_kernel[:, units:2*units]
        w_c = lstm_kernel[:, 2*units:3*units]
        w_o = lstm_kernel[:, 3*units:]

        z0 = z[:, :units]               # w_i x + w_ri h_tm1 + b_i
        z1 = z[:, units:2 * units]      # w_f x + w_rf h_tm1 + b_f
        z2 = z[:, 2 * units:3 * units]  # w_c x + w_rc h_tm1 + b_c
        z3 = z[:, 3 * units:]           # w_o x + w_ro h_tm1 + b_o

        i = model.layers[1].cell.recurrent_activation(z0)
        f = model.layers[1].cell.recurrent_activation(z1)
        c = f * c_tm1 + i * model.layers[1].cell.activation(z2)
        o = model.layers[1].cell.recurrent_activation(z3)

        _tanh = tf.keras.activations.tanh

        d_o = _tanh(c) * o * (1-o) * w_o
        d_i = i * (1-i) * w_i * _tanh(z2)
        d_f = c_tm1 * f * (1-f) * w_f
        d_c = i * (1-_tanh(z2) * _tanh(z2)) * c_tm1 * (1 - c_tm1) * w_c
        d_h = d_o + o*(1-_tanh(c)*_tanh(c))*(d_f + d_i + d_c)

        d_b = tf.keras.backend.sum(tf.keras.backend.dot(d_h, dense_kernel), axis=1, keepdims=True)

        ## calcuate loss
        # mu = 0
        # sigma = 1

        _diff = y_pred[:, 1:, :] - y_pred[:, :-1, :][0, :, :]
        diff = tf.reshape(_diff, shape=(-1,))
        _normalized_db = tf.clip_by_value(tf.abs(d_b), clip_value_min=1e-18, clip_value_max=1e18)
        normalized_db = tf.reshape(_normalized_db, shape=(-1,))[1:]

        loss1 = tf.keras.backend.square((diff - mu)/sigma) / 2.0
        loss2 = -tf.keras.backend.log(normalized_db)

        loss = loss1 + loss2
        return tf.math.reduce_sum(loss)

    def elu_loss(y_true, y_pred):
        # Extract weights from layers
        lstm_kernel = model.layers[1].cell.kernel
        lstm_recurrent_kernel = model.layers[1].cell.recurrent_kernel
        units = model.layers[1].cell.units
        h_tm1 = model.layers[1].states[0]  # previous memory state
        c_tm1 = model.layers[1].states[1]  # previous carry state

        dense_kernel = model.layers[2].kernel

        inputs = model.layers[0].output[0, :, :]

        z = tf.keras.backend.dot(inputs, lstm_kernel)
        z += tf.keras.backend.dot(h_tm1, lstm_recurrent_kernel)
        if model.layers[1].cell.use_bias:
            z = tf.keras.backend.bias_add(z, model.layers[1].cell.bias)

        w_i = lstm_kernel[:, :units]
        w_f = lstm_kernel[:, units:2*units]
        w_c = lstm_kernel[:, 2*units:3*units]
        w_o = lstm_kernel[:, 3*units:]

        z0 = z[:, :units]               # w_i x + w_ri h_tm1 + b_i
        z1 = z[:, units:2 * units]      # w_f x + w_rf h_tm1 + b_f
        z2 = z[:, 2 * units:3 * units]  # w_c x + w_rc h_tm1 + b_c
        z3 = z[:, 3 * units:]           # w_o x + w_ro h_tm1 + b_o

        i = model.layers[1].cell.recurrent_activation(z0)
        f = model.layers[1].cell.recurrent_activation(z1)
        c = f * c_tm1 + i * model.layers[1].cell.activation(z2)
        o = model.layers[1].cell.recurrent_activation(z3)

        _activation = model.layers[1].cell.activation
        _elu = tf.keras.activations.elu

        clipped_z2 = tf.clip_by_value(z2, clip_value_min=-100, clip_value_max=0)
        clipped_c = tf.clip_by_value(c, clip_value_min=-100, clip_value_max=0)

        d_o = _activation(c) * o * (1-o) * w_o
        d_i = i * (1-i) * w_i * _activation(z2)
        d_f = c_tm1 * f * (1-f) * w_f
        d_c = i * tf.keras.backend.exp(clipped_z2) * c_tm1 * (1 - c_tm1) * w_c

        d_h = d_o + o*tf.keras.backend.exp(clipped_c)*(d_f + d_i + d_c)

        d_b = tf.keras.backend.sum(tf.keras.backend.dot(d_h, dense_kernel), axis=1, keepdims=True)

        _diff = y_pred[:, 1:, :] - y_pred[:, :-1, :][0, :, :]
        diff = tf.reshape(_diff, shape=(-1,))
        _normalized_db = tf.clip_by_value(tf.abs(d_b), clip_value_min=1e-18, clip_value_max=1e18)
        normalized_db = tf.reshape(_normalized_db, shape=(-1,))[1:]

        loss1 = tf.keras.backend.square((diff - mu)/sigma) / 2.0
        loss2 = -tf.keras.backend.log(normalized_db)

        loss = loss1 + loss2
        # import ipdb; ipdb.set_trace()

        return tf.math.reduce_sum(loss)


    if activation == 'tanh':
        LOG.debug(colors.cyan("Using tanh activation"))
        return tanh_loss
    elif activation == 'elu':
        LOG.debug(colors.cyan("Using elu activation"))
        return elu_loss
    else:
        raise Exception("unknown loss function")



def lstm_mle(input_fname, units, epochs=1000, weights_fname=None, force_train=False, learning_rate=0.001, mu=0, sigma=1, activation='tanh'):

    LOG.debug(colors.cyan("Using MLE to train LSTM network..."))
    _inputs, _outputs = tdata.DatasetLoader.load_data(input_fname)
    inputs, outputs = _inputs[:2000], _outputs[:2000]
    # _train_inputs, _train_outputs = tdata.DatasetLoader.load_train_data(input_fname)
    # _test_inputs, _test_outputs = tdata.DatasetLoader.load_test_data(input_fname)
    _train_inputs, _train_outputs = inputs[:1500], outputs[:1500]
    _test_inputs, _test_outputs = inputs[1500:], outputs[1500:]

    train_inputs = _train_inputs.reshape(1, -1, 1)
    train_outputs = _train_outputs.reshape(1, -1, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    start = time.time()

    x = tf.keras.layers.Input(shape=(_train_inputs.shape[0], 1), batch_size=1)

    z, h_tm1, c_tm1 = tf.keras.layers.LSTM(int(units),
                                           input_shape=(_train_inputs.shape[0], 1),
                                           unroll=False,
                                           return_sequences=True,
                                           return_state=True,
                                           use_bias=True,
                                           stateful=True,
                                           batch_size=1,
                                           activation=activation,
                                           implementation=2)(x)
    y = tf.keras.layers.Dense(1)(z)

    model = tf.keras.models.Model(inputs=x, outputs=y)

    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    # mu = 0
    # sigma = 1
    model.compile(loss=mle_loss(model, mu, sigma, activation), optimizer=optimizer, metrics=['mse'])
    model.summary()

    if force_train or not os.path.isfile(weights_fname) :
        model.fit(train_inputs, train_outputs, epochs=epochs, verbose=1,
                  # callbacks=[early_stopping_callback],
                  # validation_split=0.05,  # need to fix bug of sample
                  batch_size=1,
                  shuffle=False)
        os.makedirs(os.path.dirname(weights_fname), exist_ok=True)
        model.save_weights(weights_fname)
    else:
        model.load_weights(weights_fname)

    end = time.time()

    LOG.debug(colors.red("time cost: {}s".format(end- start)))

    test_inputs = np.hstack([_test_inputs, np.zeros(_train_inputs.shape[0]-_test_inputs.shape[0])])
    test_inputs = test_inputs.reshape(1, -1, 1)
    test_outputs = _test_outputs.reshape(-1)

    predictions = model.predict(test_inputs)
    pred_outputs = predictions.reshape(-1)
    pred_outputs = pred_outputs[:_test_inputs.shape[0]]
    rmse = np.sqrt(np.mean((pred_outputs - test_outputs) ** 2))

    LOG.debug(colors.red("LSTM rmse: {}".format(rmse)))

    return _test_inputs, pred_outputs, rmse, end-start


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs",
                        required=False, default=100,
                        type=int)
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
    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")
    parser.add_argument('--force_train', dest='force_train',
                        required=False,
                        action="store_true")
    parser.add_argument('--markov-chain', dest='mc',
                        required=False,
                        action="store_true")

    parser.add_argument('--loss', dest='loss',
                        required=False,
                        default='mse',
                        type=str),

    parser.add_argument('--__activation__', dest='__activation__',
                        required=False,
                        default='tanh',
                        type=str),

    argv = parser.parse_args(sys.argv[1:])

    activation = argv.activation
    nb_plays = argv.nb_plays
    units = argv.units
    __units__ = argv.__units__  # 16, 32, 64, 128
    __activation__ = argv.__activation__  # 16, 32, 64, 128
    mu = int(argv.mu)
    sigma = int(argv.sigma)
    points = argv.points
    epochs = argv.epochs
    force_train = argv.force_train
    lr = argv.lr
    loss = argv.loss
    state = 0
    method = 'sin'
    input_dim = 1
    markov_chain = argv.mc
    if markov_chain is True:
        input_fname = constants.DATASET_PATH['models_diff_weights_mc_stock_model'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=1000, input_dim=input_dim, loss=loss, __activation__=__activation__)
        prediction_fname = constants.DATASET_PATH['lstm_diff_weights_mc_stock_model_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, __activation__=__activation__)
        loss_fname = constants.DATASET_PATH['lstm_diff_weights_mc_stock_model_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, __activation__=__activation__)
        weights_fname = constants.DATASET_PATH['lstm_diff_weights_mc_stock_model_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, __activation__=__activation__)

    elif argv.diff_weights is True:
        input_fname = constants.DATASET_PATH['models_diff_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, loss=loss)
        prediction_fname = constants.DATASET_PATH['lstm_diff_weights_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
        loss_fname = constants.DATASET_PATH['lstm_diff_weights_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
        weights_fname = constants.DATASET_PATH['lstm_diff_weights_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
    else:
        input_fname = constants.DATASET_PATH['models'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, loss=loss)
        prediction_fname = constants.DATASET_PATH['lstm_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
        loss_fname = constants.DATASET_PATH['lstm_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
        weights_fname = constants.DATASET_PATH['lstm_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)



    LOG.debug("====================INFO====================")
    LOG.debug(colors.cyan("units: {}".format(units)))
    LOG.debug(colors.cyan("__units__: {}".format(__units__)))
    # LOG.debug(colors.cyan("method: {}".format(method)))
    LOG.debug(colors.cyan("nb_plays: {}".format(nb_plays)))
    # LOG.debug(colors.cyan("input_dim: {}".format(input_dim)))
    # LOG.debug(colors.cyan("state: {}".format(state)))
    LOG.debug(colors.cyan("mu: {}".format(mu)))
    LOG.debug(colors.cyan("sigma: {}".format(sigma)))
    LOG.debug(colors.cyan("activation: {}".format(activation)))
    LOG.debug(colors.cyan("points: {}".format(points)))
    LOG.debug(colors.cyan("epochs: {}".format(epochs)))
    LOG.debug(colors.cyan("lr: {}".format(lr)))
    LOG.debug(colors.cyan("input file {}".format(input_fname)))
    LOG.debug(colors.cyan("prediction file {}".format(prediction_fname)))
    LOG.debug(colors.cyan("loss file {}".format(loss_fname)))
    LOG.debug(colors.cyan("loss function: {}".format(loss)))
    LOG.debug("================================================================================")

    if loss == 'mse':
        test_inputs, predictions, rmse, diff_tick = lstm(input_fname, units=__units__,
                                                         epochs=epochs, weights_fname=weights_fname,
                                                         force_train=force_train,
                                                         learning_rate=lr)
    elif loss == 'mle':
        test_inputs, predictions, rmse, diff_tick = lstm_mle(input_fname, units=__units__,
                                                             epochs=epochs, weights_fname=weights_fname,
                                                             force_train=force_train,
                                                             learning_rate=lr,
                                                             mu=mu,
                                                             sigma=sigma,
                                                             activation=__activation__)
    else:
        raise Exception("Unknown loss function")

    tdata.DatasetSaver.save_data(test_inputs, predictions, prediction_fname)
    tdata.DatasetSaver.save_loss({"rmse": float(rmse), "diff_tick": float(diff_tick)}, loss_fname)
