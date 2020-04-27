import time
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

import constants
import utils
import log as logging
import trading_data as tdata
from core import MyModel

LOG = logging.getLogger(__name__)


def hnn_predict(inputs,
                outputs,
                units=1,
                activation='tanh',
                nb_plays=1,
                weights_name='model.h5',
                ensemble=1):

    with open("{}/{}plays/input_shape.txt".format(weights_name[:-3], nb_plays), 'r') as f:
        line = f.read()
    shape = list(map(int, line.split(":")))
    assert len(shape) == 3, "shape must be 3 dimensions"

    start = time.time()

    input_dim = shape[2]
    timestep = shape[1]

    if input_dim * timestep > inputs[1].shape[0]:
        # we need to append extra value to make test_inputs and test_outpus to have the same size
        # keep test_ouputs unchange
        inputs[0] = inputs[0]
        inputs[1] = np.hstack([inputs[1], np.zeros(input_dim*timestep-inputs[1].shape[0])])

    start = time.time()
    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays,
                      parallel_prediction=True,
                      ensemble=ensemble)

    mymodel.load_weights(weights_name)
    op_outputs = mymodel.get_op_outputs_parallel(inputs[0])
    states_list = [o[-1] for o in op_outputs]
    predictions = mymodel.predict_parallel(inputs[1], states_list=states_list)

    end = time.time()
    LOG.debug("time cost: {}s".format(end-start))

    predictions = predictions[:outputs[1].shape[0]]
    loss = ((predictions - outputs[1]) ** 2).mean()
    loss = float(loss)
    LOG.debug("loss: {}".format(loss))

    return inputs[1][:outputs[1].shape[0]], predictions


def model_nb_plays_generator_with_noise(points, nb_plays, units, activation, mu, sigma, with_noise, diff_weights,
                                        __units__, __nb_plays__, __activation__, ensemble):
    # method = 'sin'
    method = 'debug-pavel'
    input_dim = 1
    state = 0
    loss = 'mse'

    if abs(sigma - int(sigma)) < 1e-5:
        sigma = int(sigma)

    input_fname = constants.DATASET_PATH['models_diff_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)
    # prediction_fname = constants.DATASET_PATH['lstm_diff_weights_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)


    # inputs, predictions = tdata.DatasetLoader.load_data(prediction_fname)
    import ipdb; ipdb.set_trace()
    _inputs, _outputs = tdata.DatasetLoader.load_data(input_fname)
    # outputs = _outputs[:, -1]
    outputs = _outputs

    from scipy.interpolate import interp1d

    interp = 10
    t_ = np.linspace(1, points, points)

    f2 = interp1d(t_, _inputs, kind='cubic')
    t_interp = np.linspace(1, points, (int)(interp*points-interp+1))

    _inputs_interp = np.interp(t_interp, t_, _inputs)
    _inputs_interp = f2(t_interp)
    clip_length = int((t_interp.shape[0] // input_dim) * input_dim)
    _inputs_interp = _inputs_interp[:clip_length]
    import ipdb; ipdb.set_trace()

    _, ground_truth_interp = tdata.DatasetGenerator.systhesis_model_generator(inputs=_inputs_interp,
                                                                              nb_plays=nb_plays,
                                                                              points=t_interp.shape[0],
                                                                              units=units,
                                                                              mu=None,
                                                                              sigma=None,
                                                                              input_dim=input_dim,
                                                                              activation=activation,
                                                                              with_noise=None,
                                                                              method=None,
                                                                              diff_weights=True,
                                                                              individual=False)
    start_pos = 6500
    end_pos = 8300

    ####### LSTM
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(__units__),
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))
    loss = 'mse'
    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    weights_fname = constants.DATASET_PATH['lstm_diff_weights_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss=loss)
    model.load_weights(weights_fname)

    test_inputs = _inputs_interp.reshape(-1, 1, 1)
    predictions_interp = model.predict(test_inputs)
    pred_outputs_interp = predictions_interp.reshape(-1)

    pred_outputs_interp = pred_outputs_interp[start_pos:end_pos]
    ############################################################################################################
    # HNN
    # weights_fname = constants.DATASET_PATH['models_diff_weights_saved_weights'].format(method=method,
    #                                                                                    activation=activation,
    #                                                                                    state=state,
    #                                                                                    mu=mu,
    #                                                                                    sigma=sigma,
    #                                                                                    units=units,
    #                                                                                    nb_plays=nb_plays,
    #                                                                                    points=points,
    #                                                                                    input_dim=input_dim,
    #                                                                                    __activation__=__activation__,
    #                                                                                    __state__=0,
    #                                                                                    __units__=__units__,
    #                                                                                    __nb_plays__=__nb_plays__,
    #                                                                                    loss='mse')

    # train_inputs = _inputs_interp[:start_pos]
    # train_outputs = ground_truth_interp[:start_pos]
    # test_inputs = _inputs_interp[start_pos:end_pos]
    # test_outputs = ground_truth_interp[start_pos:end_pos]

    # _, pred_outputs_interp = hnn_predict(inputs=[train_inputs, test_inputs],
    #                                      outputs=[train_outputs, test_outputs],
    #                                      units=__units__,
    #                                      activation=__activation__,
    #                                      nb_plays=__nb_plays__,
    #                                      weights_name=weights_fname,
    #                                      ensemble=ensemble)

    import ipdb; ipdb.set_trace()

    #########################################################
    inputs = _inputs_interp[start_pos:end_pos]
    outputs_interp = ground_truth_interp[start_pos:end_pos]

    inputs = np.vstack([inputs, inputs]).T
    outputs = np.vstack([outputs_interp, pred_outputs_interp]).T
    colors = utils.generate_colors(outputs.shape[-1])
    fname = './debug.gif'
    utils.save_animation(inputs, outputs, fname, step=40, colors=colors, mode='snake')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", dest="operator",
                        required=False,
                        action="store_true",
                        help="generate operators' dataset")
    parser.add_argument("--play", dest="play",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")
    parser.add_argument("--model", dest="model",
                        required=False,
                        action="store_true",
                        help="generate models' dataset")

    parser.add_argument("--nb_plays", dest="nb_plays",
                        required=False,
                        type=int)

    parser.add_argument("--method", dest="method",
                        required=False,
                        type=str)

    parser.add_argument("--__nb_plays__", dest="__nb_plays__",
                        required=False,
                        type=int)
    parser.add_argument("--units", dest="units",
                        required=False,
                        type=int)
    parser.add_argument("--__units__", dest="__units__",
                        required=False,
                        type=int)
    parser.add_argument("--play-noise", dest="play_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")
    parser.add_argument('--activation', dest='activation',
                        required=False,
                        default=None,
                        help='acitvation of non-linear layer')
    parser.add_argument('--__activation__', dest='__activation__',
                        required=False,
                        default=None,
                        help='acitvation of non-linear layer')
    parser.add_argument("--mc", dest="mc",
                        required=False,
                        action="store_true")
    parser.add_argument("--mu", dest="mu",
                        required=False,
                        type=float)

    parser.add_argument("--sigma", dest="sigma",
                        required=False,
                        type=float)
    parser.add_argument("--points", dest="points",
                        required=False,
                        type=int)

    parser.add_argument("--ensemble", dest="ensemble",
                        required=False,
                        type=int)

    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")

    parser.add_argument("--model-noise", dest="model_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")

    parser.add_argument("--with-noise", dest="with_noise",
                        required=False,
                        action="store_true",
                        help="generate plays' dataset")

    argv = parser.parse_args(sys.argv[1:])

    model_nb_plays_generator_with_noise(argv.points,
                                        argv.nb_plays,
                                        argv.units,
                                        argv.activation,
                                        int(argv.mu),
                                        float(argv.sigma),
                                        with_noise=argv.with_noise,
                                        diff_weights=argv.diff_weights,
                                        __units__=argv.__units__,
                                        __activation__=argv.__activation__,
                                        __nb_plays__=argv.__nb_plays__,
                                        ensemble=argv.ensemble)
