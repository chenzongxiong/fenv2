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
def gru(input_fname, units, epochs=1000, weights_fname=None, force_train=False, learning_rate=0.001):

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
    model.add(tf.keras.layers.GRU(int(units),
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


# def mle_loss(model, mu, sigma, activation='tanh', debug=False):
#     def tanh_helper(y_true, y_pred):
#         # Extract weights from layers
#         lstm_kernel = model.layers[1].cell.kernel
#         lstm_recurrent_kernel = model.layers[1].cell.recurrent_kernel
#         units = model.layers[1].cell.units
#         h_tm1 = model.layers[1].states[0]  # previous memory state
#         c_tm1 = model.layers[1].states[1]  # previous carry state

#         dense_kernel = model.layers[2].kernel

#         inputs = model.layers[0].output[0, :, :]

#         z = tf.keras.backend.dot(inputs, lstm_kernel)
#         z += tf.keras.backend.dot(h_tm1, lstm_recurrent_kernel)
#         if model.layers[1].cell.use_bias:
#             z = tf.keras.backend.bias_add(z, model.layers[1].cell.bias)

#         w_i = lstm_kernel[:, :units]
#         w_f = lstm_kernel[:, units:2*units]
#         w_c = lstm_kernel[:, 2*units:3*units]
#         w_o = lstm_kernel[:, 3*units:]

#         z0 = z[:, :units]               # w_i x + w_ri h_tm1 + b_i
#         z1 = z[:, units:2 * units]      # w_f x + w_rf h_tm1 + b_f
#         z2 = z[:, 2 * units:3 * units]  # w_c x + w_rc h_tm1 + b_c
#         z3 = z[:, 3 * units:]           # w_o x + w_ro h_tm1 + b_o

#         _tanh = tf.keras.activations.tanh

#         i = model.layers[1].cell.recurrent_activation(z0)
#         f = model.layers[1].cell.recurrent_activation(z1)
#         c_tilde = _tanh(z2)
#         c = f * c_tm1 + i * c_tilde
#         o = model.layers[1].cell.recurrent_activation(z3)



#         d_o = _tanh(c) * o * (1-o) * w_o
#         d_i = i * (1-i) * w_i * c_tilde
#         d_f = c_tm1 * f * (1-f) * w_f
#         # d_c = i * (1-_tanh(z2)) * (1+_tanh(z2)) * c_tm1 * (1 - c_tm1) * w_c
#         d_c = i * (1-c_tilde) * (1+c_tilde) * w_c
#         d_h = d_o + o*(1-_tanh(c))*(1+_tanh(c))*(d_f + d_i + d_c)

#         d_b = tf.keras.backend.sum(tf.keras.backend.dot(d_h, dense_kernel), axis=1, keepdims=True)

#         ## calcuate loss
#         # mu = 0
#         # sigma = 1

#         _diff = y_pred[0, 1:, 0] - y_pred[0, :-1, 0]
#         diff = tf.reshape(_diff, shape=(-1,))
#         _normalized_db = tf.clip_by_value(tf.abs(d_b), clip_value_min=1e-18, clip_value_max=1e18)
#         normalized_db = tf.reshape(_normalized_db, shape=(-1,))[1:]

#         loss1 = tf.keras.backend.square((diff - mu)/sigma) / 2.0
#         loss2 = -tf.keras.backend.log(normalized_db)

#         loss = loss1 + loss2
#         # return tf.math.reduce_sum(loss)
#         return tf.math.reduce_sum(loss), tf.math.reduce_sum(loss1), tf.math.reduce_sum(loss2), tf.math.reduce_sum(d_o), tf.math.reduce_sum(d_i), tf.math.reduce_sum(d_c), tf.math.reduce_sum(d_h), tf.math.reduce_sum(d_b), tf.math.reduce_sum(d_f)

#     # def elu_loss(y_true, y_pred):
#     #     # Extract weights from layers
#     #     lstm_kernel = model.layers[1].cell.kernel
#     #     lstm_recurrent_kernel = model.layers[1].cell.recurrent_kernel
#     #     units = model.layers[1].cell.units
#     #     h_tm1 = model.layers[1].states[0]  # previous memory state
#     #     c_tm1 = model.layers[1].states[1]  # previous carry state

#     #     dense_kernel = model.layers[2].kernel

#     #     inputs = model.layers[0].output[0, :, :]

#     #     z = tf.keras.backend.dot(inputs, lstm_kernel)
#     #     z += tf.keras.backend.dot(h_tm1, lstm_recurrent_kernel)
#     #     if model.layers[1].cell.use_bias:
#     #         z = tf.keras.backend.bias_add(z, model.layers[1].cell.bias)

#     #     w_i = lstm_kernel[:, :units]
#     #     w_f = lstm_kernel[:, units:2*units]
#     #     w_c = lstm_kernel[:, 2*units:3*units]
#     #     w_o = lstm_kernel[:, 3*units:]

#     #     z0 = z[:, :units]               # w_i x + w_ri h_tm1 + b_i
#     #     z1 = z[:, units:2 * units]      # w_f x + w_rf h_tm1 + b_f
#     #     z2 = z[:, 2 * units:3 * units]  # w_c x + w_rc h_tm1 + b_c
#     #     z3 = z[:, 3 * units:]           # w_o x + w_ro h_tm1 + b_o


#     #     i = model.layers[1].cell.recurrent_activation(z0)
#     #     f = model.layers[1].cell.recurrent_activation(z1)
#     #     c = f * c_tm1 + i * model.layers[1].cell.activation(z2)
#     #     o = model.layers[1].cell.recurrent_activation(z3)

#     #     _activation = model.layers[1].cell.activation
#     #     _elu = tf.keras.activations.elu

#     #     clipped_z2 = tf.clip_by_value(z2, clip_value_min=-18, clip_value_max=0)
#     #     # clipped_z2 = tf.clip_by_value(z2, clip_value_min=0)

#     #     clipped_c = tf.clip_by_value(c, clip_value_min=-18, clip_value_max=0)

#     #     d_o = _activation(c) * o * (1-o) * w_o
#     #     d_i = i * (1-i) * w_i * _activation(z2)
#     #     d_f = c_tm1 * f * (1-f) * w_f
#     #     d_c = i * tf.keras.backend.exp(clipped_z2) * c_tm1 * (1 - c_tm1) * w_c

#     #     d_h = d_o + o*tf.keras.backend.exp(clipped_c)*(d_f + d_i + d_c)

#     #     d_b = tf.keras.backend.sum(tf.keras.backend.dot(d_h, dense_kernel), axis=1, keepdims=True)

#     #     _diff = y_pred[0, 1:, 0] - y_pred[0, :-1, 0]
#     #     diff = tf.reshape(_diff, shape=(-1,))
#     #     _normalized_db = tf.clip_by_value(tf.keras.backend.square(d_b), clip_value_min=1e-18, clip_value_max=1e18)
#     #     normalized_db = tf.reshape(_normalized_db, shape=(-1,))[1:]

#     #     loss1 = tf.keras.backend.square((diff - mu)/sigma)
#     #     # loss1 = tf.clip_by_value(loss1, clip_value_min=1e-9, clip_value_max=1e9)
#     #     loss2 = -tf.keras.backend.log(normalized_db)

#     #     loss = (loss1 + loss2)/2.0
#     #     import ipdb; ipdb.set_trace()

#     #     return tf.math.reduce_sum(loss)

#     def elu_helper(y_true, y_pred):
#         # Extract weights from layers
#         lstm_kernel = model.layers[1].cell.kernel
#         lstm_recurrent_kernel = model.layers[1].cell.recurrent_kernel
#         units = model.layers[1].cell.units
#         h_tm1 = model.layers[1].states[0]  # previous memory state
#         c_tm1 = model.layers[1].states[1]  # previous carry state

#         dense_kernel = model.layers[2].kernel

#         inputs = model.layers[0].output[0, :, :]

#         z = tf.keras.backend.dot(inputs, lstm_kernel)
#         z += tf.keras.backend.dot(h_tm1, lstm_recurrent_kernel)
#         if model.layers[1].cell.use_bias:
#             z = tf.keras.backend.bias_add(z, model.layers[1].cell.bias)

#         w_i = lstm_kernel[:, :units]
#         w_f = lstm_kernel[:, units:2*units]
#         w_c = lstm_kernel[:, 2*units:3*units]
#         w_o = lstm_kernel[:, 3*units:]

#         z0 = z[:, :units]               # w_i x + w_ri h_tm1 + b_i
#         z1 = z[:, units:2 * units]      # w_f x + w_rf h_tm1 + b_f
#         z2 = z[:, 2 * units:3 * units]  # w_c x + w_rc h_tm1 + b_c
#         z3 = z[:, 3 * units:]           # w_o x + w_ro h_tm1 + b_o

#         _elu = tf.keras.activations.elu
#         # _tanh = tf.keras.activations.tanh

#         i = model.layers[1].cell.recurrent_activation(z0)
#         f = model.layers[1].cell.recurrent_activation(z1)
#         c_tilde = _elu(z2)
#         c = f * c_tm1 + i * c_tilde
#         o = model.layers[1].cell.recurrent_activation(z3)

#         # i = model.layers[1].cell.recurrent_activation(z0)
#         # f = model.layers[1].cell.recurrent_activation(z1)
#         # c = f * c_tm1 + i * model.layers[1].cell.activation(z2)
#         # o = model.layers[1].cell.recurrent_activation(z3)

#         # _activation = model.layers[1].cell.activation

#         clipped_z2 = tf.clip_by_value(z2, clip_value_min=-9, clip_value_max=0)
#         # clipped_z2 = tf.clip_by_value(z2, clip_value_min=0)

#         clipped_c = tf.clip_by_value(c, clip_value_min=-9, clip_value_max=0)

#         d_o = _elu(c) * o * (1-o) * w_o
#         d_i = i * (1-i) * w_i * c_tilde
#         d_f = c_tm1 * f * (1-f) * w_f
#         d_c = i * tf.keras.backend.exp(clipped_z2) * w_c

#         d_h = d_o + o*tf.keras.backend.exp(clipped_c)*(d_f + d_i + d_c)

#         d_b = tf.keras.backend.sum(tf.keras.backend.dot(d_h, dense_kernel), axis=1, keepdims=True)

#         _diff = y_pred[0, 1:, 0] - y_pred[0, :-1, 0]
#         diff = tf.reshape(_diff, shape=(-1,))
#         _normalized_db = tf.clip_by_value(tf.keras.backend.square(d_b), clip_value_min=1e-18, clip_value_max=1e18)
#         normalized_db = tf.reshape(_normalized_db, shape=(-1,))[1:]

#         loss1 = tf.keras.backend.square((diff - mu)/sigma)
#         loss1 = tf.clip_by_value(loss1, clip_value_min=1e-18, clip_value_max=1e18)
#         loss2 = -tf.keras.backend.log(normalized_db)

#         loss = (loss1 + loss2)/2.0

#         return tf.math.reduce_sum(loss), tf.math.reduce_sum(loss1), tf.math.reduce_sum(loss2), tf.math.reduce_sum(d_o), tf.math.reduce_sum(d_i), tf.math.reduce_sum(d_c), tf.math.reduce_sum(d_h), tf.math.reduce_sum(d_b), tf.math.reduce_sum(d_f)

#     def debug_loss(y_true, y_pred):
#         # return elu_helper(y_true, y_pred)[0]
#         # import ipdb; ipdb.set_trace()
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[0]

#     def debug_loss1(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)

#         return mylocals[func_name](y_true, y_pred)[1]
#         # return elu_helper(y_true, y_pred)[1]

#     def debug_loss2(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[2]
#         # return elu_helper(y_true, y_pred)[1]

#         return elu_helper(y_true, y_pred)[2]

#     def debug_d_o(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[3]
#         # return elu_helper(y_true, y_pred)[1]

#         return elu_helper(y_true, y_pred)[3]

#     def debug_d_i(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[4]
#         # return elu_helper(y_true, y_pred)[1]

#         return elu_helper(y_true, y_pred)[4]

#     def debug_d_c(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[5]
#         # return elu_helper(y_true, y_pred)[1]

#         return elu_helper(y_true, y_pred)[5]

#     def debug_d_h(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[6]
#         # return elu_helper(y_true, y_pred)[1]

#         return elu_helper(y_true, y_pred)[6]

#     def debug_d_b(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[7]
#         # return elu_helper(y_true, y_pred)[1]


#     def debug_d_f(y_true, y_pred):
#         func_name = '{}_helper'.format(activation)
#         return mylocals[func_name](y_true, y_pred)[8]
#         # return elu_helper(y_true, y_pred)[1]

#         return elu_helper(y_true, y_pred)[8]

#     # if debug == 'loss' and activation == 'elu':
#     #     return debug_loss

#     # if debug == 'loss1' and activation == 'elu':
#     #     return debug_loss1

#     # if debug == 'loss2' and activation == 'elu':
#     #     return debug_loss2


#     # if debug == 'd_o' and activation == 'elu':
#     #     return debug_d_o
#     # if debug == 'd_h' and activation == 'elu':
#     #     return debug_d_h
#     # if debug == 'd_i' and activation == 'elu':
#     #     return debug_d_i

#     # if debug == 'd_c' and activation == 'elu':
#     #     return debug_d_c
#     # if debug == 'd_b' and activation == 'elu':
#     #     return debug_d_b
#     # if debug == 'd_f' and activation == 'elu':
#     #     return debug_d_f
#     mylocals = locals()
#     if debug == 'loss':
#         return debug_loss

#     if debug == 'loss1':
#         return debug_loss1

#     if debug == 'loss2':
#         return debug_loss2


#     if debug == 'd_o':
#         return debug_d_o
#     if debug == 'd_h':
#         return debug_d_h
#     if debug == 'd_i':
#         return debug_d_i

#     if debug == 'd_c':
#         return debug_d_c
#     if debug == 'd_b':
#         return debug_d_b
#     if debug == 'd_f':
#         return debug_d_f



#     if activation == 'tanh':
#         LOG.debug(colors.cyan("Using tanh activation"))
#         return debug_loss
#     elif activation == 'elu':
#         LOG.debug(colors.cyan("Using elu activation"))
#         # return elu_loss
#         return debug_loss
#     else:
#         raise Exception("unknown loss function")



# def lstm_mle(input_fname, units, epochs=1000, weights_fname=None, force_train=False, learning_rate=0.001, mu=0, sigma=1, activation='tanh'):

#     LOG.debug(colors.cyan("Using MLE to train LSTM network, mu: {}, sigma: {}...".format(mu, sigma)))
#     inputs, outputs = tdata.DatasetLoader.load_data(input_fname)
#     inputs, outputs = inputs[:1700], outputs[:1700]
#     _train_inputs, _train_outputs = inputs[:1300], outputs[:1300]
#     _test_inputs, _test_outputs = inputs[1300:], outputs[1300:]

#     # inputs, outputs = inputs[:1000], outputs[:1000]
#     # _train_inputs, _train_outputs = inputs[:600], outputs[:600]
#     # _test_inputs, _test_outputs = inputs[600:], outputs[600:]

#     train_inputs = _train_inputs.reshape(1, -1, 1)
#     train_outputs = _train_outputs.reshape(1, -1, 1)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#     # early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
#     start = time.time()

#     x = tf.keras.layers.Input(shape=(_train_inputs.shape[0], 1), batch_size=1)

#     z, h_tm1, c_tm1 = tf.keras.layers.LSTM(int(units),
#                                            input_shape=(_train_inputs.shape[0], 1),
#                                            unroll=False,
#                                            return_sequences=True,
#                                            return_state=True,
#                                            use_bias=True,
#                                            stateful=True,
#                                            batch_size=1,
#                                            activation=activation,
#                                            implementation=2)(x)
#     y = tf.keras.layers.Dense(1)(z)

#     model = tf.keras.models.Model(inputs=x, outputs=y)

#     # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
#     # mu = 0
#     # sigma = 1
#     model.compile(loss=mle_loss(model, mu, sigma, activation), optimizer=optimizer, metrics=['mse',
#                                                                                              mle_loss(model, mu, sigma, activation, 'loss'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'loss1'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'loss2'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'd_o'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'd_i'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'd_c'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'd_h'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'd_b'),
#                                                                                              mle_loss(model, mu, sigma, activation, 'd_f'),

#                 ])

#     model.summary()

#     if force_train or not os.path.isfile(weights_fname) :
#         model.fit(train_inputs, train_outputs, epochs=epochs, verbose=1,
#                   # callbacks=[early_stopping_callback],
#                   # validation_split=0.05,  # need to fix bug of sample
#                   batch_size=1,
#                   shuffle=False)
#         os.makedirs(os.path.dirname(weights_fname), exist_ok=True)
#         model.save_weights(weights_fname)
#     else:
#         model.load_weights(weights_fname)

#     end = time.time()

#     LOG.debug(colors.red("time cost: {}s".format(end- start)))

#     test_inputs = np.hstack([_test_inputs, np.zeros(_train_inputs.shape[0]-_test_inputs.shape[0])])
#     test_inputs = test_inputs.reshape(1, -1, 1)
#     test_outputs = _test_outputs.reshape(-1)

#     predictions = model.predict(test_inputs)
#     pred_outputs = predictions.reshape(-1)
#     pred_outputs = pred_outputs[:_test_inputs.shape[0]]
#     rmse = np.sqrt(np.mean((pred_outputs - test_outputs) ** 2))

#     LOG.debug(colors.red("LSTM rmse: {}".format(rmse)))

#     return _test_inputs, pred_outputs, rmse, end-start


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
                        default=0,
                        type=float)
    parser.add_argument("--__sigma__", dest="__sigma__",
                        required=False,
                        default=0,
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

    parser.add_argument('--method', dest='method',
                        required=False,
                        default='sin',
                        type=str),

    parser.add_argument('--ensemble', dest='ensemble',
                        required=False,
                        default=1,
                        type=int),

    argv = parser.parse_args(sys.argv[1:])

    activation = argv.activation
    nb_plays = argv.nb_plays
    units = argv.units
    __units__ = argv.__units__  # 16, 32, 64, 128
    __activation__ = argv.__activation__  # 16, 32, 64, 128
    mu = 0
    sigma = argv.sigma
    __sigma__ = argv.__sigma__
    if sigma == int(sigma):
        sigma = int(sigma)
    if __sigma__ == int(__sigma__):
        __sigma__ = int(__sigma__)

    points = argv.points
    epochs = argv.epochs
    force_train = argv.force_train
    lr = argv.lr
    loss = argv.loss
    state = 0
    method = argv.method
    ensemble = argv.ensemble

    input_dim = 1
    # markov_chain = argv.mc
    # if markov_chain is True:
    #     input_fname = constants.DATASET_PATH['models_diff_weights_mc_stock_model'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=1000, input_dim=input_dim, loss=loss, __activation__=__activation__)
    #     prediction_fname = constants.DATASET_PATH['lstm_diff_weights_mc_stock_model_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, __activation__=__activation__)
    #     loss_fname = constants.DATASET_PATH['lstm_diff_weights_mc_stock_model_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, __activation__=__activation__)
    #     weights_fname = constants.DATASET_PATH['lstm_diff_weights_mc_stock_model_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, __activation__=__activation__)

    # el
    if argv.diff_weights is True:
        input_fname = constants.DATASET_PATH['models_diff_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, loss=loss)
        prediction_fname = constants.DATASET_PATH['gru_diff_weights_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, ensemble=ensemble)
        loss_fname = constants.DATASET_PATH['gru_diff_weights_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, ensemble=ensemble)
        weights_fname = constants.DATASET_PATH['gru_diff_weights_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss, ensemble=ensemble)
    # else:
    #     input_fname = constants.DATASET_PATH['models'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, loss=loss)
    #     prediction_fname = constants.DATASET_PATH['lstm_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
    #     loss_fname = constants.DATASET_PATH['lstm_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
    #     weights_fname = constants.DATASET_PATH['lstm_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)


    LOG.debug("====================INFO====================")
    LOG.debug(colors.cyan("units: {}".format(units)))
    LOG.debug(colors.cyan("__units__: {}".format(__units__)))
    LOG.debug(colors.cyan("method: {}".format(method)))
    LOG.debug(colors.cyan("nb_plays: {}".format(nb_plays)))
    # LOG.debug(colors.cyan("input_dim: {}".format(input_dim)))
    # LOG.debug(colors.cyan("state: {}".format(state)))
    LOG.debug(colors.cyan("mu: {}".format(mu)))
    LOG.debug(colors.cyan("sigma: {}".format(sigma)))
    LOG.debug(colors.cyan("__sigma__: {}".format(__sigma__)))
    LOG.debug(colors.cyan("activation: {}".format(activation)))
    LOG.debug(colors.cyan("__activation__: {}".format(__activation__)))
    LOG.debug(colors.cyan("points: {}".format(points)))
    LOG.debug(colors.cyan("epochs: {}".format(epochs)))
    LOG.debug(colors.cyan("lr: {}".format(lr)))
    LOG.debug(colors.cyan("input file {}".format(input_fname)))
    LOG.debug(colors.cyan("prediction file {}".format(prediction_fname)))
    LOG.debug(colors.cyan("loss file {}".format(loss_fname)))
    LOG.debug(colors.cyan("loss function: {}".format(loss)))
    LOG.debug("================================================================================")

    if loss == 'mse':
        test_inputs, predictions, rmse, diff_tick = gru(input_fname, units=__units__,
                                                        epochs=epochs, weights_fname=weights_fname,
                                                        force_train=force_train,
                                                        learning_rate=lr)
    # elif loss == 'mle':
    #     test_inputs, predictions, rmse, diff_tick = lstm_mle(input_fname, units=__units__,
    #                                                          epochs=epochs, weights_fname=weights_fname,
    #                                                          force_train=force_train,
    #                                                          learning_rate=lr,
    #                                                          mu=mu,
    #                                                          sigma=__sigma__,
    #                                                          activation=__activation__)
    # else:
    #     raise Exception("Unknown loss function")

    tdata.DatasetSaver.save_data(test_inputs, predictions, prediction_fname)
    tdata.DatasetSaver.save_loss({"rmse": float(rmse), "diff_tick": float(diff_tick)}, loss_fname)
    LOG.debug('==================FINISHED=========================================================')
