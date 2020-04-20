import os
import time
import copy
import inspect
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
MP_CONTEXT = mp.get_context('spawn')

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers import Dense

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import training_arrays
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops

import numpy as np

import utils
import colors
import constants
import trading_data as tdata
import log as logging


LOG = logging.getLogger(__name__)

sess = utils.get_session()
session = utils.get_session()
SESS = utils.get_session()
SESSION = utils.get_session()

once = True
SENTINEL = utils.sentinel_marker()

tf.keras.backend.set_epsilon(1e-9)


hacking = 1


def do_guess_helper(step, direction, start_price, nb_plays, activation, sign, prev_states, weights, delta=0.001):
    '''
    Parameters:
    --------------------
    step: current attemping step
    direction: in which direction the price should change
    start_price: where the price start to do guessing
    nb_plays: the number of plays attend in this transaction
    activation: the type of activation used during prediction, can be 'None', 'tanh', 'relu'
    sign: -1/+1, to correct the result of noise
    prev_states: a list of previous state, see the architecture of Operator
    weights: the weights of this trained neural network
    delta: the minimum step should increase to find a root
    Returns:
    --------------------
    guess_price: the price guess, scalar
    predict_noise: the noise correpsonding to guess price, scalar
    '''
    predict_noise_list = []
    guess = start_price + direction * step * delta
    # LOG.debug("--------------------------------------------------------------------------------");
    # LOG.debug("prev_states: {}".format(prev_states))
    for i in range(nb_plays):
        prev_state = prev_states[i]
        p = phi(weights[0][i] * guess - prev_state) + prev_state
        pp = weights[1][i] * p + weights[2][i]
        if activation is None:
            pass
        elif activation == 'tanh':
            pp = np.tanh(pp)
        elif activation == 'relu':
            pp =  pp * (pp > 0)
        elif activation == 'elu':
            pp1 = (pp >= 0) * pp
            pp2 = (pp < 0) * pp
            pp2 = np.exp(pp2) - 1
            pp = pp1 + pp2
        elif activation == 'softmax':
            pp = np.log(1 + np.exp(pp))
        else:
            raise Exception("not support for activation {}".format(colors.red(activation)))

        ppp = (weights[3][i] * pp).sum() + weights[4][i]
        predict_noise_list.append(ppp[0])

    predict_noise = sign * sum(predict_noise_list) / nb_plays
    return guess, predict_noise


def do_guess_seq(start,
                 seq,
                 prev_gt_price,
                 curr_gt_price,
                 prev_gt_prediction,
                 curr_gt_prediction,
                 mu,
                 sigma,
                 nb_plays,
                 activation,
                 sign,
                 individual_p_list,
                 weights,
                 hysteresis_info,
                 max_iteration=2000):
    '''
    Parameters:
    --------------------
    start: the start position of the prices going to do prediction
    seq: the length of prices going to do prediction
    prev_gt_price: previous ground-truth price
    curr_gt_price: current ground-truth price
    prev_gt_noise: previous ground-truth noise
    curr_gt_noise: current ground-truth noise
    mu: the empirical mean of prediction dataset(noise set)
    sigma: the empirical standard derivation of prediction dataset(noise set)
    nb_plays: the number of plays attend in this transaction
    activation: the type of activation used during prediction, can be 'None', 'tanh', 'relu'
    sign: -1/+1, to correct the result of noise
    individual_p_list: a list of operator outputs, mainly we need to track the previous state
    weights: the weights of this trained neural network
    hysteresis_info: information collected for ploting the internal behaviours during each transaction
    max_iteration: the max iterations during trying to find a root for the price
    Returns:
    --------------------
    guess_price_seq: a list of guess price, the length of it is equal to `seq`
    '''
    logger_string1 = "Step: {}, true_price: {:.5f}, guess price: {:.5f}, guess noise: {:.5f}, generated noise: {:.5f}, true noise: {:.5f}, prev true noise: {:.5f}, curr_diff: {:.5f}, prev_diff: {:.5f}, direction: {}, delta: {}, mu: {}, sigma {}"
    logger_string2 = "Step: {}, true_price: {:.5f}, guess price: {:.5f}, guess noise: {:.5f}, generated noise: {:.5f}, true noise: {:.5f}, curr_diff: {:.5f}, prev_diff: {:.5f}, direction: {}, delta: {}, mu: {}, sigma: {}"
    delta = 0.001
    ####################################################################################################
    # guess_price_seq: the first value in it is the start point of price in prediction                 #
    # predict_noise_seq: the first value in it is the start point of gt-noise in prediction            #
    ####################################################################################################
    individual_p_list = copy.deepcopy(individual_p_list)
    LOG.debug("Before do_guess_helper, len(individual_p_list): {}, len(individual_p_list[0]): {}".format(len(individual_p_list), len(individual_p_list[0])))

    guess_price_seq = [prev_gt_price]
    predict_noise_seq = [prev_gt_prediction]
    interval = 0


    while interval < seq:
        k = start + interval
        bk = np.random.normal(loc=mu, scale=sigma) + predict_noise_seq[-1]

        # bk = curr_gt_prediction
        # always predict noise at [-sigma] and [sigma]
        # global hacking
        # bk = 110 * hacking
        # hacking = - hacking
        if bk > predict_noise_seq[-1]:
            direction = -1
        elif bk < predict_noise_seq[-1]:
            direction = +1
        else:
            direction = 0

        step = 0
        guess_hysteresis_list = [(guess_price_seq[-1], predict_noise_seq[-1])]

        prev_states = [individual_p_list[i][-1] for i in range(nb_plays)]
        guess, guess_noise = do_guess_helper(step, direction, guess_price_seq[-1], nb_plays, activation, sign, prev_states, weights)

        if np.allclose(predict_noise_seq[-1], guess_noise) is False:
            # sanity checking
            LOG.error("predict_noise_seq[-1] is: {}, guess_noise is: {}, they should be the same".format(predict_noise_seq[-1], guess_noise)),
            # import ipdb; ipdb.set_trace()

        prev_diff, curr_diff = None, guess_noise - bk
        good_guess = False
        while step <= max_iteration:
            step += 1
            prev_diff = curr_diff
            guess, guess_noise = do_guess_helper(step, direction, guess_price_seq[-1], nb_plays, activation, sign, prev_states, weights)

            guess_hysteresis_list.append((guess, guess_noise))
            curr_diff = guess_noise - bk
            if np.abs(curr_diff) < 0.01 or curr_diff * prev_diff < 0:
                LOG.debug(colors.yellow(logger_string1.format(step, float(curr_gt_price), float(guess), float(guess_noise), float(bk), float(curr_gt_prediction), float(prev_gt_prediction), float(curr_diff), float(prev_diff), direction,
                                                              delta, mu, sigma)))
                good_guess = True
                break

            LOG.debug(logger_string2.format(step, float(curr_gt_price), float(guess), float(guess_noise), float(bk), float(curr_gt_prediction), float(curr_diff), float(prev_diff), direction, delta, mu, sigma))

        #########################################################################################################################
        # hysteresis_info:                                                                                                      #
        #   guess_hysteresis_list    -> hysteresis_info[0]: a list of (price, noise) tuples                                     #
        #   original_prediction[k-1] -> hysteresis_info[1]: the ground truth of noise in previous step                          #
        #   original_prediction[k]   -> hysteresis_info[2]: the ground truth of noise in current step                           #
        #   bk                       -> hysteresis_info[3]: the noise generated from random walk                                #
        #   price[k-1]               -> hysteresis_info[4]: the ground truth of price in previous step                          #
        #   price[k]                 -> hysteresis_info[5]: the ground truth of price in current step                           #
        #########################################################################################################################
        hysteresis_info.append([guess_hysteresis_list, prev_gt_prediction, curr_gt_prediction, bk, prev_gt_price, curr_gt_price])

        if good_guess is False:
            LOG.warn(colors.red("Not a good guess"))
            continue

        guess_price_seq.append(guess)
        predict_noise_seq.append(guess_noise)

        for j in range(nb_plays):
            p = phi(weights[0][j] * guess - individual_p_list[j][-1]) + individual_p_list[j][-1]
            individual_p_list[j].append(p)

        interval += 1

        LOG.debug("After do_guess_helper, len(individual_p_list): {}, len(individual_p_list[0]): {}".format(len(individual_p_list), len(individual_p_list[0])))
    return guess_price_seq[1:], bk


def repeat(k,
           seq,
           repeating,
           prev_gt_price,
           curr_gt_price,
           prev_gt_prediction,
           curr_gt_prediction,
           mu,
           sigma,
           nb_plays,
           activation,
           sign,
           operator_outputs,
           weights,
           ensemble,
           real_mu,
           real_sigma):
    '''
    Parameters:
    --------------------
    k: the index of the price starting to prediction
    seq: the sequence of prices trying to predict
    repeating: how many times this sequence should repeat
    prev_gt_price: previous ground-truth price
    curr_gt_price: current ground-truth price
    prev_gt_noise: previous ground-truth noise
    curr_gt_noise: current ground-truth noise
    mu: the empirical mean of prediction dataset(noise set)
    sigma: the empirical standard derivation of prediction dataset(noise set)
    nb_plays: the number of plays attend in this transaction
    activation: the type of activation used during prediction, can be 'None', 'tanh', 'relu'
    sign: -1/+1, to correct the result of noise
    operator_outputs: a list of operator outputs, mainly we need to track the previous state
    weights: the weights of this trained neural network
    Returns:
    --------------------
    guess_price_seq: the avearge of this guess price sequence.
    '''
    logger_string3 = "================ Guess k: {} successfully, predict price: {:.5f}, grouth-truth price: {:.5f} prev gt price: {:.5f}, std: {:.5f} ====================="

    hysteresis_info = []
    guess_price_seq_stack = []

    individual_p_list = [[o[k-1]] for o in operator_outputs]
    bk_list = []
    for _ in range(repeating):
        guess_price_seq, bk = do_guess_seq(k,
                                           seq,
                                           prev_gt_price,
                                           curr_gt_price,
                                           prev_gt_prediction,
                                           curr_gt_prediction,
                                           mu,
                                           sigma,
                                           nb_plays,
                                           activation,
                                           sign,
                                           individual_p_list,
                                           weights,
                                           hysteresis_info)
        bk_list.append(bk)
        guess_price_seq_stack.append(guess_price_seq)

    guess_price_seq_stack_ = np.array(guess_price_seq_stack)
    bk_list_ = np.array(bk_list)
    # LOG.debug("guess_price_seq_stack_: {}".format(guess_price_seq_stack_))
    # LOG.debug("guess_price_seq_stack_.shape: {}".format(guess_price_seq_stack_.shape))
    # LOG.debug("guess_price_seq_stack_.mean(): {}".format(guess_price_seq_stack_.mean()))
    # LOG.debug("guess_price_seq_stack_.std(): {}".format(guess_price_seq_stack_.std()))

    avg_guess = guess_price_seq_stack_.mean(axis=0)[-1]
    LOG.debug(colors.red(logger_string3.format(k, float(avg_guess), float(curr_gt_price), float(prev_gt_price), float(guess_price_seq_stack_.std()))))
    LOG.debug("********************************************************************************")
    utils.plot_internal_transaction(hysteresis_info, k, predicted_price=float(avg_guess), mu=real_mu, sigma=real_sigma, guess_price_seq=guess_price_seq_stack_, bk_list=bk_list_, ensemble=ensemble)

    return avg_guess


def wrapper_repeat(args):
    k = args[0]
    seq = args[1]
    repeating = args[2]
    prev_gt_price = args[3]
    curr_gt_price = args[4]
    prev_gt_prediction = args[5]
    curr_gt_prediction = args[6]
    mu = args[7]
    sigma = args[8]
    nb_plays = args[9]
    activation = args[10]
    sign = args[11]
    operator_outputs = args[12]
    weights = args[13]

    ensemble = args[14]
    real_mu = args[15]
    real_sigma = args[16]

    return repeat(k,
                  seq,
                  repeating,
                  prev_gt_price,
                  curr_gt_price,
                  prev_gt_prediction,
                  curr_gt_prediction,
                  mu,
                  sigma,
                  nb_plays,
                  activation,
                  sign,
                  operator_outputs,
                  weights,
                  ensemble,
                  real_mu,
                  real_sigma)


def phi(x, width=1.0):
    if x[0] > (width/2.0):
        return (x[0] - width/2.0)
    elif x[0] < (-width/2.0):
        return (x[0] + width/2.0)
    else:
        return float(0)


def Phi(x, width=1.0, weight=1.0):
    '''
    Phi(x) = x - width/2 , if x > width/2
           = x + width/2 , if x < - width/2
           = 0         , otherwise
    '''
    assert x.shape[0].value == 1 and x.shape[1].value == 1, "x must be a scalar"

    ZEROS = tf.zeros(x.shape, dtype=tf.float32, name='zeros')
    # _width = tf.constant([[width/2.0]], dtype=tf.float32)
    _width = tf.constant([[width/2.0]], dtype=tf.float32)
    r1 = tf.cond(tf.reduce_all(tf.less(x, -_width)), lambda: weight*(x + _width), lambda: ZEROS)
    r2 = tf.cond(tf.reduce_all(tf.greater(x, _width)), lambda: weight*(x - _width), lambda: r1)
    return r2


def LeakyPhi(x, width=1.0, alpha=0.001):
    '''
    TODO:
    LeakyPhi(x) = x - width/2 , if x > width/2
                = x + width/2 , if x < - width/2
                = alpha         , otherwise
    '''
    assert x.shape[0].value == 1 and x.shape[1].value == 1, "x must be a scalar"

    ZEROS = tf.zeros(x.shape, dtype=tf.float32, name='zeros')
    # _width = tf.constant([[width/2.0]], dtype=tf.float32)
    _width = tf.constant([[width/2.0]], dtype=tf.float32)
    r1 = tf.cond(tf.reduce_all(tf.less(x, -_width)), lambda: x + _width, lambda: ZEROS)
    r2 = tf.cond(tf.reduce_all(tf.greater(x, _width)), lambda: x - _width, lambda: r1)
    return r2


def gradient_operator(P, weights=None):
    reshaped_P = tf.reshape(P, shape=(P.shape[0].value, -1))

    diff_ = reshaped_P[:, 1:] - reshaped_P[:, :-1]
    x0 = tf.slice(reshaped_P, [0, 0], [1, 1])
    diff_ = tf.concat([x0, diff_], axis=1)
    result = tf.cast(tf.abs(diff_) >= 1e-9, dtype=tf.float32)

    return tf.reshape(result * weights, shape=P.shape)


def jacobian(outputs, inputs):
    jacobian_matrix = []
    M = outputs.shape[1].value
    for m in range(M):
        # We iterate over the M elements of the output vector
        grad_func = tf.gradients(outputs[0, m, 0], inputs)[0]
        jacobian_matrix.append(tf.reshape(grad_func, shape=(M, )))

    return ops.convert_to_tensor(jacobian_matrix, dtype=tf.float32)


def gradient_nonlinear_layer(fZ, weights=None, activation=None, reduce_sum=True):
    LOG.debug("gradient nonlinear activation {}".format(colors.red(activation)))

    _fZ = tf.reshape(fZ, shape=fZ.shape.as_list()[1:])

    if activation is None:
        partial_gradient = tf.keras.backend.ones(shape=_fZ.shape)
    elif activation == 'tanh':
        partial_gradient = (1.0 + _fZ) * (1.0 - _fZ)
    elif activation == 'relu':
        partial_gradient = tf.cast(_fZ >= 1e-9, dtype=tf.float32)
    elif activation == 'elu':
        a = tf.cast(_fZ >= 0, dtype=tf.float32)
        b = tf.cast(_fZ < 0, dtype=tf.float32)
        partial_gradient = a + b * (_fZ + 1)
    elif activation == 'softmax':
        partial_gradient = 1.0 - 1.0 / tf.math.exp(fZ)
    else:
        raise Exception("activation: {} not support".format(activation))

    if reduce_sum is True:
        gradient = tf.reduce_sum(partial_gradient * weights, axis=-1, keepdims=True)
    else:
        gradient = partial_gradient * weights

    g = tf.reshape(gradient, shape=(fZ.shape.as_list()[:-1] + [gradient.shape[-1].value]))
    return g


def gradient_linear_layer(weights, multiples=1, expand_dims=True):
    if expand_dims is True:
        return tf.expand_dims(tf.tile(tf.transpose(weights, perm=[1, 0]), multiples=[multiples, 1]), axis=0)
    else:
        return tf.tile(tf.transpose(weights, perm=[1, 0]), multiples=[multiples, 1])


def gradient_operator_nonlinear_layers(P,
                                       fZ,
                                       operator_weights,
                                       nonlinear_weights,
                                       activation,
                                       debug=False,
                                       inputs=None,
                                       reduce_sum=True,
                                       feed_dict=dict()):

    if debug is True and inputs is not None:
        LOG.debug(colors.red("Only use under unittest, not for real situation"))
        J = jacobian(P, inputs)
        g1 = tf.reshape(tf.reduce_sum(J, axis=0), shape=inputs.shape)
        calc_g = gradient_operator(P, operator_weights)

        utils.init_tf_variables()
        J_result, calc_g_result = session.run([J, calc_g], feed_dict=feed_dict)
        if not np.allclose(np.diag(J_result), calc_g_result.reshape(-1)):
            LOG.error(colors.red("ERROR: gradient operator- and nonlinear- layers"))
            import ipdb; ipdb.set_trace()
    else:
        g1 = gradient_operator(P, operator_weights)

    g1 = tf.reshape(g1, shape=P.shape)
    g2 = gradient_nonlinear_layer(fZ, nonlinear_weights, activation, reduce_sum=reduce_sum)
    g3 = g1*g2
    return g3


def gradient_all_layers(P,
                        fZ,
                        operator_weights,
                        nonlinear_weights,
                        linear_weights,
                        activation,
                        debug=False,
                        inputs=None,
                        feed_dict=dict()):

    g1 = gradient_operator_nonlinear_layers(P, fZ,
                                            operator_weights,
                                            nonlinear_weights,
                                            activation=activation,
                                            debug=debug,
                                            inputs=inputs,
                                            reduce_sum=False,
                                            feed_dict=feed_dict)
    g2 = tf.expand_dims(tf.matmul(g1[0], linear_weights), axis=0)
    return g2


def confusion_matrix(y_true, y_pred, labels=['increase', 'descrease']):
    '''
    # https://en.wikipedia.org/wiki/Confusion_matrix
    confusion matrix is impolemented as following
                        | increase (truth)  | descrease (truth) |
        increase (pred) |                   |                   |
       descrease (pred) |                   |                   |
    '''

    diff1 = ((y_true[1:] - y_true[:-1]) >= 0).tolist()
    diff2 = ((y_pred[1:] - y_true[:-1]) >= 0).tolist()
    confusion = np.zeros(4, dtype=np.int32)
    for a, b in zip(diff1, diff2):
        if a is True and b is True:
            confusion[0] += 1
        elif a is True and b is False:
            confusion[2] += 1
        elif a is False and b is True:
            confusion[1] += 1
        elif a is False and b is False:
            confusion[3] += 1

    return confusion.reshape([2, 2])


class PhiCell(Layer):
    def __init__(self,
                 input_dim=1,
                 weight=1.0,
                 width=1.0,
                 hysteretic_func=Phi,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint="non_neg",
                 **kwargs):
        self.debug = kwargs.pop("debug", False)
        self.unroll = kwargs.pop('unittest', False)

        super(PhiCell, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self._weight = weight
        self._recurrent_weight = -1
        self._width = width
        self.units = 1
        self.state_size = [1]

        self.kernel_initializer = tf.keras.initializers.Constant(value=weight, dtype=tf.float32)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.hysteretic_func = hysteretic_func
        self.input_dim = input_dim
        self._timestep = 0

    def build(self, input_shape):
        if self.debug:
            LOG.debug("Initialize *weight* as pre-defined: {} ....".format(self._weight))
            self.kernel = tf.Variable([[self._weight]], name="weight", dtype=tf.float32)
        else:
            LOG.debug("Initialize *weight* randomly...")
            assert self.units == 1, "Phi Cell unit must be equal to 1"

            self.kernel = self.add_weight(
                "weight",
                shape=(1, 1),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=tf.float32,
                trainable=True)

        # self._state = self.add_weight(
        #     "state",
        #     shape=(1, 1),
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     dtype=tf.float32,
        #     trainable=True)

        self.built = True

    def call(self, inputs, states):
        """
        Parameters:
        ----------------
        inputs: `inputs` is a vector, the shape of inputs vector is like [1 * sequence length]
                Here, we consider the length of sequence is the same as the batch-size.
        state: `state` is randomly initialized, the shape of is [1 * 1]
        """
        self._inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)
        # self._state = ops.convert_to_tensor(states[-1], dtype=tf.float32)
        self._state = states[-1]

        LOG.debug("PhiCellinputs.shape: {}".format(inputs.shape))
        LOG.debug("PhiCell._inputs.shape: {}".format(self._inputs.shape))
        LOG.debug("PhiCell._state.shape: {}".format(self._state.shape))
        ############### IMPL from Scratch #####################
        # outputs_ = tf.multiply(self._inputs, self.kernel)
        # outputs = [self._state]
        # for i in range(outputs_.shape[-1].value):
        #     output = tf.add(Phi(tf.subtract(outputs_[0][i], outputs[-1]), self._width, 0.99), outputs[-1])
        #     outputs.append(output)

        # outputs = ops.convert_to_tensor(outputs[1:], dtype=tf.float32)
        # state = outputs[-1]
        # outputs = tf.reshape(outputs, shape=self._inputs.shape)

        # LOG.debug("before reshaping state.shape: {}".format(state.shape))
        # state = tf.reshape(state, shape=(-1, 1))
        # LOG.debug("after reshaping state.shape: {}".format(state.shape))

        # return outputs, [state]

        ################ IMPL via RNN ###########################
        def inner_steps(inputs, states):
            # import ipdb; ipdb.set_trace()
            print("inner timestep: ", self._timestep)
            self._timestep += 1

            LOG.debug("inputs: {}, states: {}".format(inputs, states))
            outputs = Phi(inputs - states[-1], self._width, 1.0) + states[-1]
            return outputs, [outputs]

        # import ipdb; ipdb.set_trace()
        self._inputs = tf.multiply(self._inputs, self.kernel)
        inputs_ = tf.reshape(self._inputs, shape=(1, self._inputs.shape[0].value*self._inputs.shape[1].value, 1))
        if isinstance(states, list) or isinstance(states, tuple):
            self._states = ops.convert_to_tensor(states[-1], dtype=tf.float32)
        else:
            self._states = ops.convert_to_tensor(states, dtype=tf.float32)

        assert self._state.shape.ndims == 2, colors.red("PhiCell states must be 2 dimensions")
        # states_ = [tf.reshape(self._states, shape=self._states.shape.as_list())]
        states_ = states
        # import ipdb; ipdb.set_trace()
        last_outputs_, outputs_, states_x = tf.keras.backend.rnn(inner_steps, inputs=inputs_, initial_states=states_, unroll=self.unroll)
        LOG.debug("outputs_.shape: {}".format(outputs_))
        LOG.debug("states_x.shape: {}".format(states_x))
        print("timestep: ", self._timestep)
        # import ipdb; ipdb.set_trace()
        return outputs_, list(states_x)


class Operator(RNN):
    def __init__(self,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 unittest=False):

        cell = PhiCell(
            weight=weight,
            width=1.0,
            debug=debug,
            unittest=unittest)

        unroll = True if unittest is True else False
        super(Operator, self).__init__(
            cell=cell,
            return_sequences=True,
            return_state=False,
            stateful=True,
            unroll=False)

    def build(self, input_shape):
        # import ipdb; ipdb.set_trace()
        self._states = [self.add_weight(
            name="states",
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(value=0.0,  dtype=tf.float32),
            trainable=True,
            shape=(1, 1))]

        super(Operator, self).build(input_shape)
        # if self.stateful:
        #     self.reset_states()
        # self.built = True

    def call(self, inputs, initial_state=None):
        LOG.debug("Operator.inputs.shape: {}".format(inputs.shape))
        output = super(Operator, self).call(inputs, initial_state=initial_state)
        assert inputs.shape.ndims == 3, colors.red("ERROR: Input from Operator must be 3 dimensions")
        shape = inputs.shape.as_list()
        output_ = tf.reshape(output, shape=(shape[0], -1, 1))
        return output_

    @property
    def kernel(self):
        return self.cell.kernel

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the batch size by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            self.states = [
                           tf.keras.backend.zeros([batch_size] + tensor_shape.as_shape(dim).as_list())
                           for dim in self.cell.state_size
                           ]
        elif states is None:
            # import ipdb; ipdb.set_trace()
            for state, dim in zip(self.states, self.cell.state_size):
                tf.keras.backend.set_value(state,
                                           np.zeros([batch_size] +
                                                    tensor_shape.as_shape(dim).as_list()))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' + str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                dim = self.cell.state_size[index]
                if value.shape != tuple([batch_size] +
                                        tensor_shape.as_shape(dim).as_list()):
                    raise ValueError(
                        'State ' + str(index) + ' is incompatible with layer ' +
                        self.name + ': expected shape=' + str(
                            (batch_size, dim)) + ', found shape=' + str(value.shape))
                # TODO(fchollet): consider batch calls to `set_value`.
                tf.keras.backend.set_value(state, value)


def my_softmax(x):
    return tf.math.log(1 + tf.math.exp(x))


class MyDense(Layer):
    def __init__(self, units=1,
                 activation="tanh",
                 weight=1,
                 use_bias=True,
                 activity_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        self._debug = kwargs.pop("debug", False)
        if '_init_kernel' in kwargs:
            self._init_kernel = kwargs.pop("_init_kernel")
        # self._init_kernel = 1.0
        self._init_bias = kwargs.pop("_init_bias", 0)

        super(MyDense, self).__init__(**kwargs)
        self.units = units
        self._weight = weight

        if activation == 'softmax':
            self.activation = my_softmax
        else:
            self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        if use_bias is True:
            self.bias_initializer = initializers.get(bias_initializer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.bias_constraint = constraints.get(bias_constraint)
        self.use_bias = use_bias

    def build(self, input_shape):
        if self._debug:
            LOG.debug("init mydense kernel/bias as pre-defined")
            if hasattr(self, '_init_kernel'):
                _init_kernel = np.array([[self._init_kernel for i in range(self.units)]])
            else:
                _init_kernel = np.random.uniform(low=0.0, high=1.5, size=self.units)
            _init_kernel = _init_kernel.reshape([1, -1])
            LOG.debug(colors.yellow("kernel: {}".format(_init_kernel)))
            self.kernel = tf.Variable(_init_kernel, name="theta", dtype=tf.float32)

            if self.use_bias is True:
                # _init_bias = 0
                _init_bias = self._init_bias
                LOG.debug(colors.yellow("bias: {}".format(_init_bias)))

                self.bias = tf.Variable(_init_bias, name="bias", dtype=tf.float32)
        else:
            self.kernel = self.add_weight(
                "theta",
                shape=(1, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=tf.float32,
                trainable=True)

            if self.use_bias:
                self.bias = self.add_weight(
                    "bias",
                    shape=(1, self.units),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    dtype=tf.float32,
                    trainable=True)

        self.built = True

    def call(self, inputs):
        assert inputs.shape.ndims == 3

        # XXX: double checked. it's correct in current model. no worried
        outputs = inputs * self.kernel
        if self.use_bias:
            outputs += self.bias

        if self.activation is not None:
            outputs =  self.activation(outputs)

        return outputs


class MySimpleDense(Dense):
    def __init__(self, **kwargs):
        self._debug = kwargs.pop("debug", False)
        self._init_bias = kwargs.pop("_init_bias", 0)
        if '_init_kernel' in kwargs:
            self._init_kernel = kwargs.pop("_init_kernel")
        # self._init_kernel = 1.0
        kwargs['activation'] = None
        kwargs['kernel_constraint'] = None
        super(MySimpleDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert self.units == 1
        if self._debug is True:
            LOG.debug("init mysimpledense kernel/bias as pre-defined")
            if hasattr(self, '_init_kernel'):
                _init_kernel = np.array([self._init_kernel for i in range(input_shape[-1].value)])
            else:
                _init_kernel = np.random.uniform(low=0.0, high=1.5, size=input_shape[-1].value)
            _init_kernel = _init_kernel.reshape(-1, 1)
            LOG.debug(colors.yellow("kernel: {}".format(_init_kernel)))

            self.kernel = tf.Variable(_init_kernel, name="kernel", dtype=tf.float32)

            if self.use_bias:
                _init_bias = (self._init_bias,)
                LOG.debug(colors.yellow("bias: {}".format(_init_bias)))
                self.bias = tf.Variable(_init_bias, name="bias", dtype=tf.float32)
        else:
            super(MySimpleDense, self).build(input_shape)

        self.built = True

    def call(self, inputs):
        return super(MySimpleDense, self).call(inputs)


class Play(object):
    def __init__(self,
                 inputs=None,
                 units=1,
                 batch_size=1,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 activation="tanh",
                 loss="mse",
                 optimizer="adam",
                 network_type=constants.NetworkType.OPERATOR,
                 use_bias=True,
                 name="play",
                 timestep=1,
                 input_dim=1,
                 **kwargs):

        np.random.seed(kwargs.pop("ensemble", 1))

        self._weight = weight
        self._width = width
        self._debug = debug

        self.activation = activation

        self.loss = loss
        self.optimizer = optimizer

        self._play_timestep = timestep
        self._play_batch_size = batch_size
        self._play_input_dim = input_dim

        self.units = units

        self._network_type = network_type
        self.built = False
        self._need_compile = False
        self.use_bias = use_bias
        self._name = name
        self._unittest = kwargs.pop('unittest', False)

    def _make_batch_input_shape(self, inputs=None):
        self._batch_input_shape = tf.TensorShape([1, self._play_timestep, self._play_input_dim])

    def build(self, inputs=None):
        if inputs is None and self._batch_input_shape is None:
            raise Exception("Unknown input shape")
        if inputs is not None:
            _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)

            if _inputs.shape.ndims == 1:
                length = _inputs.shape[-1].value
                if length % (self._play_input_dim * self._play_timestep) != 0:
                    LOG.error("length is: {}, input_dim: {}, play_timestep: {}".format(length,
                                                                                       self._play_input_dim,
                                                                                       self._play_timestep))
                    raise Exception("The batch size cannot be divided by the length of input sequence.")

                # self.batch_size = length // (self._play_timestep * self._play_input_dim)
                self._play_batch_size = length // (self._play_timestep * self._play_input_dim)
                self.batch_size = 1
                self._batch_input_shape = tf.TensorShape([self.batch_size, self._play_timestep, self._play_input_dim])

            else:
                raise Exception("dimension of inputs must be equal to 1")

        length = self._batch_input_shape[1].value * self._batch_input_shape[2].value
        self.batch_size = self._batch_input_shape[0].value
        assert self.batch_size == 1, colors.red("only support batch_size is 1")
        if not getattr(self, "_unittest", False):
            assert self._play_timestep == 1, colors.red("only support outter-timestep 1")

        self.model = tf.keras.models.Sequential()

        CACHE = utils.get_cache()
        input_layer = CACHE.get('play_input_layer', None)

        if input_layer is None:
            input_layer = tf.keras.layers.InputLayer(batch_size=self.batch_size,
                                                     input_shape=self._batch_input_shape[1:])
            CACHE['play_input_layer'] = input_layer

        self.model.add(input_layer)

        self.model.add(Operator(weight=getattr(self, "_weight", 1.0),
                                width=getattr(self, "_width", 1.0),
                                debug=getattr(self, "_debug", False),
                                unittest=getattr(self, "_unittest", False)))

        if self._network_type == constants.NetworkType.PLAY:
            self.model.add(MyDense(self.units,
                                   activation=self.activation,
                                   use_bias=self.use_bias,
                                   debug=getattr(self, "_debug", False)))

            self.model.add(MySimpleDense(units=1,
                                         activation=None,
                                         use_bias=True,
                                         debug=getattr(self, "_debug", False)))

        if self._need_compile is True:
            LOG.info(colors.yellow("Start to compile this model"))
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=[self.loss])

        self._early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)
        self._tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=constants.LOG_DIR,
                                                                     histogram_freq=0,
                                                                     batch_size=self.batch_size,
                                                                     write_graph=True,
                                                                     write_grads=False,
                                                                     write_images=False)

        # if not getattr(self, "_preload_weights", False):
        #     utils.init_tf_variables()
        if self._network_type == constants.NetworkType.OPERATOR:
            LOG.debug(colors.yellow("SUMMARY of Operator"))
        elif self._network_type == constants.NetworkType.PLAY:
            LOG.debug(colors.yellow("SUMMARY of {}".format(self._name)))
        else:
            raise
        self.model.summary()
        self.built = True

    def reshape(self, inputs, outputs=None):
        LOG.debug("reshape inputs to: {}".format(self._batch_input_shape))
        x = tf.reshape(inputs, shape=self._batch_input_shape)
        if outputs is not None:
            if self._network_type == constants.NetworkType.OPERATOR:
                y = tf.reshape(outputs, shape=(self._batch_input_shape[0].value, -1, 1))
            elif self._network_type == constants.NetworkType.PLAY:
                y = tf.reshape(outputs, shape=(self._batch_input_shape[0].value, -1, 1))
            return x, y
        else:
            return x

    def fit(self, inputs, outputs, epochs=100, verbose=0, steps_per_epoch=1):
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        outputs = ops.convert_to_tensor(outputs, tf.float32)
        self._need_compile = True
        if not self.built:
            self.build(inputs)

        x, y = self.reshape(inputs, outputs)

        self.model.fit(x,
                       y,
                       epochs=epochs,
                       verbose=verbose,
                       steps_per_epoch=steps_per_epoch,
                       batch_size=None,
                       shuffle=False,
                       callbacks=[self._early_stopping_callback,
                                  self._tensor_board_callback])

    def evaluate(self, inputs, outputs, steps_per_epoch=1):
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        outputs = ops.convert_to_tensor(outputs, tf.float32)

        if not self.built:
            self.build(inputs)

        x, y = self.reshape(inputs, outputs)
        return self.model.evaluate(x, y, steps=steps_per_epoch)

    def predict(self, inputs, steps_per_epoch=1, verbose=0, states=None):
        if not self.built:
            self.build(inputs)

        outputs = []
        input_dim = self._batch_input_shape[-1].value
        samples = inputs.shape[-1] // input_dim
        LOG.debug("#Samples: {}".format(samples))
        for j in range(samples):
            # LOG.debug("PID: {}, self.states: {}, states: {} before".format(os.getpid(), sess.run(self.states), states))
            self.reset_states(states)
            # LOG.debug("PID: {}, self.states: {}, states: {} after".format(os.getpid(), sess.run(self.states), states))
            x = inputs[j*input_dim:(j+1)*input_dim].reshape(1, 1, -1)
            output = self.model.predict(x, steps=steps_per_epoch, verbose=verbose).reshape(-1)
            # LOG.debug("PID: {}, self.states: {}, states: {} done".format(os.getpid(), sess.run(self.states), states))
            outputs.append(output)
            if j != samples - 1:
                op_output = self.operator_output(x, states)
                states = op_output[-1].reshape(1, 1)

        return np.hstack(outputs)

    @property
    def states(self):
        return self.operator_layer.states

    @property
    def weights(self):
        if self._network_type == constants.NetworkType.OPERATOR:
            weights_ = [self.operator_layer.kernel]
        elif self._network_type == constants.NetworkType.PLAY:
            weights_ = [self.operator_layer.kernel,
                        self.nonlinear_layer.kernel,
                        self.nonlinear_layer.bias,
                        self.linear_layer.kernel,
                        self.linear_layer.bias]
        else:
            raise Exception("Unknown NetworkType. It must be in [OPERATOR, PLAY]")

        weights_ = utils.get_session().run(weights_)
        weights_ = [w.reshape(-1) for w in weights_]
        return weights_

    def operator_output(self, inputs, states=None):
        if len(inputs.shape) == 1:
            input_dim = self._batch_input_shape[-1].value
            samples = inputs.shape[-1] // input_dim
        elif list(inputs.shape) == self._batch_input_shape.as_list():
            input_dim = inputs.shape[-1]
            samples = inputs.shape[0]
        else:
            raise Exception("Unknown input.shape: {}".format(inputs.shape))

        outputs = []
        for j in range(samples):
            self.reset_states(states)
            x = inputs[j*input_dim:(j+1)*input_dim].reshape(1, 1, -1)
            op_output = self.operator_layer(ops.convert_to_tensor(x, dtype=tf.float32))
            output = utils.get_session().run(op_output).reshape(-1)
            outputs.append(output)
            states = output[-1].reshape(1, 1)

        outputs = np.hstack(outputs)
        return outputs.reshape(-1)

    def reset_states(self, states=None):
        self.operator_layer.reset_states(states)

    @property
    def number_of_layers(self):
        if not self.built:
            raise Exception("Model has not been built.")

        return len(self.model._layers)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path, by_name=False):
        self.model.load_weights(path, by_name=by_name)
        # self.model.load_weights(path, by_name=True)

    @property
    def layers(self):
        if hasattr(self, 'model'):
            return self.model.layers
        return []

    @property
    def _layers(self):
        if hasattr(self, 'model'):
            return self.model.layers
        return []

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def output(self):
        return self.model._layers[-1].output

    @property
    def input(self):
        return self.model._layers[0].input

    @property
    def state_updates(self):
        return self.model.state_updates

    @property
    def operator_layer(self):
        return self.layers[0]

    @property
    def nonlinear_layer(self):
        return self.layers[1]

    @property
    def linear_layer(self):
        return self.layers[2]

    def __getstate__(self):
        LOG.debug("PID: {} pickle {}".format(os.getpid(), self._name))
        if not hasattr(self, '_batch_input_shape'):
            raise Exception("_batch_input_shape must be added before pickling")

        state = {
            "_weight": self._weight,
            "_width": self._width,
            "_debug": self._debug,
            "activation": self.activation,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "_play_timestep": self._play_timestep,
            "_play_batch_size": self._play_batch_size,
            "_play_input_dim": self._play_input_dim,
            "units": self.units,
            "_network_type": self._network_type,
            "_built": getattr(self, "_built", False),
            "_need_compile": self._need_compile,
            "use_bias": self.use_bias,
            "_name": self._name,
            "_weights_fname": getattr(self, '_weights_fname', None),
            "_preload_weights": getattr(self, '_preload_weights', False),
            "_batch_input_shape": getattr(self, '_batch_input_shape'),
        }
        return state

    def __setstate__(self, d):
        LOG.debug("PID: {}, unpickle {}".format(os.getpid(), colors.cyan(d)))
        self.__dict__ = d
        if self._built is False:
            self.build()
        if self._preload_weights is False and self._weights_fname is not None:
            self._preload_weights = True
            self.load_weights(self._weights_fname)
            LOG.debug(colors.cyan("Set weights to play in sub-process"))

        LOG.debug("PID: {}, self: {}, self.model: {}".format(os.getpid(), self, self.model))

    def __hash__(self):
        return hash(self._name)


class Task(object):

    def __init__(self,
                 play=None,
                 func=None,
                 func_args=None):
        self.play = play
        self.func = func
        self.func_args = func_args

        if play is None:
            self.id = SENTINEL
        elif play._name.startswith('play-'):
            self.id = int(play._name.split('-')[-1])
        else:
            raise Exception("the format of play name isn't compatiable")

    def __getstate__(self):
        if self.play is None:
            return {
                'id': self.id
            }

        play_state = self.play.__getstate__()
        return {
            'id': self.id,
            'play': play_state,
            'func': self.func,
            'func_args': self.func_args
        }

    def __setstate__(self, d):
        cache = utils.get_cache()
        self.__dict__ = d
        if self.id == SENTINEL:
            return

        play_state = copy.deepcopy(d['play'])
        self.play = cache.get(play_state['_name'], None)

        if self.play is None:
            self.play = Play()
            self.play.__setstate__(play_state)
            cache[play_state['_name']] = self.play
            LOG.debug("Init play in sub-process")
        else:
            LOG.debug("Reuse play inside Cache.")

    def __hash__(self):
        return self.play.__hash__()


def runner(queue, results):
    '''
    queue: JoinableQueue, all tasks will put inside it
    results: Manager().dict(), all the return of task will be put inside it carefully
    '''
    # NOTE: use sentinel task to exit infinite loop
    while True:
        LOG.debug("PID: {} try to retrieve task".format(os.getpid()))
        task = queue.get()
        if task.id == SENTINEL:
            LOG.debug("PID: {} retrieve SENTINEL task: {}, break infinite loop".format(os.getpid(), task))
            break

        LOG.debug("PID: {} retrieve task: {} successfully".format(os.getpid(), task))
        attr = getattr(task.play, task.func)
        if inspect.ismethod(attr):
            result = attr(*task.func_args)
        else:
            result = attr

        results[task.play._name] = result
        queue.task_done()

    tf.keras.backend.clear_session()
    utils.get_cache().clear()


class WorkerPool(object):
    def __init__(self, pool_size=None):
        self.pool_size = constants.CPU_COUNTS if pool_size is None else pool_size
        LOG.debug("Create a worker pool with size {}".format(self.pool_size))
        self._results = mp.Manager().dict()
        self.queues = [MP_CONTEXT.JoinableQueue() for _ in range(self.pool_size)]
        self.pool = [MP_CONTEXT.Process(target=runner, args=(self.queues[i], self._results,)) for i in range(self.pool_size)]
        self._task_done = False
        self._task_sequences = []

    def start(self):
        for p in self.pool:
            p.start()

    def put(self, task):
        index = task.id % self.pool_size
        LOG.debug("PID: {} put task: {} into queue: {}.".format(os.getpid(), task, index))
        self.queues[index].put(task)
        self._task_sequences.append(task.play._name)

    def is_alive(self):
        for p in self.pool:
            if p.is_alive() is False:
                return False
        return True

    def join(self):
        self._task_done = False
        for i in range(self.pool_size):
            self.queues[i].join()
        self._task_done = True

    def close(self):
        for q in self.queues:
            q.put(Task())

    @property
    def results(self):
        if self._task_done is False:
            raise Exception("Must call `join` before retrieving results or wait until all tasks are done.")
        LOG.debug("self._task_sequences: {}".format(self._task_sequences))
        _results = [self._results[k] for k in self._task_sequences]
        self._task_sequences.clear()
        self._results.clear()
        return _results


class MyModel(Layer):
    def __init__(self,
                 nb_plays=1,
                 units=1,
                 batch_size=1,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 activation='tanh',
                 optimizer='adam',
                 timestep=1,
                 input_dim=1,
                 diff_weights=False,
                 network_type=constants.NetworkType.PLAY,
                 learning_rate=0.001,
                 ensemble=1,
                 learnable_mu=False,
                 learnable_sigma=False,
                 parallel_prediction=False,
                 **kwargs):
        super(Layer, self).__init__(**kwargs)
        self._unittest = kwargs.pop('unittest', False)
        if self._unittest is False:
            assert timestep == 1, colors.red('timestep must be 1')

        assert activation in [None, 'tanh', 'relu', 'elu', 'softmax'], colors.red("activation {} not support".format(activation))

        # fix random seed to 123
        # seed = 123
        np.random.seed(ensemble)
        LOG.debug(colors.red("Make sure you are using the right random seed. currently seed is {}".format(ensemble)))

        self.plays = []
        self._nb_plays = nb_plays
        self._units = units
        self._activation = activation
        self._input_dim = input_dim
        self._ensemble = ensemble
        _weight = 1.0
        _width = 0.1
        width = 1
        for nb_play in range(nb_plays):
            if diff_weights is True:
                weight = 0.5 / (_width * (1 + nb_play)) # width range from (0.1, ... 0.1 * nb_plays)
                # weight = 0.5 / (_width * (1 + nb_play)) # width range from (0.1, ... 0.1 * nb_plays)
                # weight = nb_play # width range from (0.1, ... 0.1 * nb_plays)
            else:
                weight = 1.0

            weight = 2 * (nb_play + 1)                  # width range from (0.1, ... 0.1 * nb_plays)
            LOG.debug("MyModel {} generates {} with Weight: {}".format(self._ensemble, colors.red("Play #{}".format(nb_play+1)), weight))
            # if debug is True:
            #     weight = 1.0

            play = Play(units=units,
                        batch_size=batch_size,
                        weight=weight,
                        width=width,
                        debug=debug,
                        activation=activation,
                        loss=None,
                        optimizer=None,
                        network_type=network_type,
                        name="play-{}".format(nb_play),
                        timestep=timestep,
                        input_dim=input_dim,
                        unittest=self._unittest,
                        ensemble=self._ensemble)
            assert play._need_compile == False, colors.red("Play inside MyModel mustn't be compiled")
            self.plays.append(play)

        # self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
        # if kwargs.pop("parallel_prediction", False):
        if parallel_prediction:
            self.parallel_prediction = True
            self.pool = WorkerPool(constants.CPU_COUNTS)
            self.pool.start()

        self._learnable_mu = learnable_mu
        self._learnable_sigma = learnable_sigma

    def fit(self,
            inputs,
            outputs,
            epochs=100,
            verbose=0,
            steps_per_epoch=1,
            loss_file_name="./tmp/mymodel_loss_history.csv",
            learning_rate=0.001,
            decay=0.):

        writer = utils.get_tf_summary_writer("./log/mse")

        inputs = ops.convert_to_tensor(inputs, tf.float32)
        __mu__ = (outputs[1:] - outputs[:-1]).mean()
        __sigma__ = (outputs[1:] - outputs[:-1]).std()
        outputs = ops.convert_to_tensor(outputs, tf.float32)

        for play in self.plays:
            if not play.built:
                play.build(inputs)

        _xs = []
        _ys = []
        for play in self.plays:
            _x, _y = play.reshape(inputs, outputs)
            _xs.append(_x)
            _ys.append(_y)

        _xs = [_x]

        params_list = []
        model_outputs = []
        feed_inputs = [utils.get_cache()['play_input_layer'].input]
        feed_targets = []

        for idx, play in enumerate(self.plays):
            # feed_inputs.append(play.input)
            model_outputs.append(play.output)

            shape = tf.keras.backend.int_shape(play.output)
            name = 'play{}_target'.format(idx)
            target = tf.keras.backend.placeholder(
                ndim=len(shape),
                name=name,
                dtype=tf.float32)
            feed_targets.append(target)

            # update_inputs += play.model.get_updates_for(inputs)
            params_list += play.trainable_weights

        if self._nb_plays > 1:
            y_pred = tf.keras.layers.Average()(model_outputs)
        else:
            y_pred = model_outputs[0]

        loss = tf.keras.backend.mean(tf.math.square(y_pred - _y))
        mu = tf.keras.backend.mean(y_pred[:, 1:, :] - y_pred[:, :-1, :])
        sigma = tf.keras.backend.std(y_pred[:, 1:, :] - y_pred[:, :-1, :])

        # decay: decay learning rate to half every 100 steps
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay)
        with tf.name_scope('training'):
            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=params_list,
                                                     loss=loss)
            # updates += update_inputs

            training_inputs = feed_inputs + feed_targets
            train_function = tf.keras.backend.function(training_inputs,
                                                       [loss, mu, sigma],
                                                       updates=updates)

        # _x = [x for _ in range(self._nb_plays)]
        # _y = [y for _ in range(self._nb_plays)]
        self._batch_input_shape = utils.get_cache()['play_input_layer'].input.shape

        ins = _xs + _ys

        self.cost_history = []

        path = "/".join(loss_file_name.split("/")[:-1])
        writer.add_graph(tf.get_default_graph())
        loss_summary = tf.summary.scalar("loss", loss)
        for i in range(epochs):
            for j in range(steps_per_epoch):
                cost, predicted_mu, predicted_sigma = train_function(ins)
            self.cost_history.append([i, cost])
            LOG.debug("Epoch: {}, Loss: {:.7f}, predicted_mu: {:.7f}, predicted_sigma: {:.7f}, truth_mu: {:.7f}, truth_sigma: {:.7f}".format(i,
                                                                                                                                             float(cost),
                                                                                                                                             float(predicted_mu),
                                                                                                                                             float(predicted_sigma),
                                                                                                                                             float(__mu__),
                                                                                                                                             float(__sigma__)))

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict_parallel(self, inputs, individual=False, states_list=None):
        _inputs = inputs
        if states_list is None:
            states_list = [np.array([0]).reshape(1, 1) for _ in range(self._nb_plays)]
        elif not isinstance(states_list[0], np.ndarray):
            states_list = [np.array([s]).reshape(1, 1) for s in states_list]
        else:
            states_list = [s.reshape(1, 1) for s in states_list]


        ##########################################################################
        #     multiprocessing.Pool
        ##########################################################################
        # pool = MP_CONTEXT .Pool(constants.CPU_COUNTS)
        # args_list = [(play, _inputs) for play in self.plays]
        # outputs = pool.map(parallel_predict, args_list)
        # pool.close()
        # pool.join()
        ##########################################################################
        ##########################################################################
        #     myImplementation.Pool
        ##########################################################################
        start = time.time()
        for i, play in enumerate(self.plays):
            LOG.debug("{}".format(play._name))
            task = Task(play, 'predict', (_inputs, 1, 0, states_list[i]))
            # task = Task(play, 'predict', (_inputs, 1, 0, None))
            self.pool.put(task)

        self.pool.join()
        end = time.time()
        outputs = self.pool.results
        LOG.debug("Cost time {} s".format(end-start))
        ##########################################################################
        #  Serial execution
        ##########################################################################
        # inputs = ops.convert_to_tensor(inputs, tf.float32)
        # x = self.plays[0].reshape(inputs)
        # for play in self.plays:
        #     if not play.built:
        #         play.build(inputs)

        # outputs = []
        # for play in self.plays:
        #     start = time.time()
        #     outputs.append(play.predict(x))
        #     end = time.time()
        #     LOG.debug("play {} cost time {} s".format(play._name, end-start))
        ##########################################################################

        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        if individual is True:
            outputs_ = outputs_.reshape(len(self.plays), -1).T
            # sanity checking
            for i in range(len(self.plays)):
                if np.all(outputs_[i, :] == outputs[i]) is False:
                    raise
            return prediction, outputs_
        return prediction

    @property
    def weights(self):
        for play in self.plays:
            LOG.debug("{}, number of layer is: {}".format(play._name, play.number_of_layers))
            LOG.debug("{}, weight: {}".format(play._name, play.weights))

    def _make_batch_input_shape(self, inputs):
        _ = [play._make_batch_input_shape(inputs) for play in self.plays if not play.built]

    def compile(self, inputs, **kwargs):
        mu, sigma = kwargs.pop('mu', None), kwargs.pop('sigma', None)
        outputs = kwargs.pop('outputs', None)
        LOG.debug("Compile with mu: {}, sigma: {}".format(colors.red(mu), colors.red(sigma)))

        _inputs = inputs
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        _ = [play.build(inputs) for play in self.plays if not play.built]

        self.params_list = []
        self.feed_inputs = []
        self.model_outputs = []

        self.feed_mu = tf.constant(mu, name='input-mu', dtype=tf.float32)
        self.feed_sigma= tf.constant(sigma, name='input-sigma', dtype=tf.float32)

        self.mu_placeholder = tf.keras.backend.placeholder(ndim=0,
                                                           name='target-mu',
                                                           dtype=tf.float32)
        self.sigma_placeholder = tf.keras.backend.placeholder(ndim=0,
                                                              name='target-sigma',
                                                              dtype=tf.float32)

        self._batch_input_shape = utils.get_cache()['play_input_layer'].input.shape
        self.feed_inputs = [utils.get_cache()['play_input_layer'].input]
        self.feed_targets = []
        self.update_inputs = []

        for play in self.plays:
            self.model_outputs.append(play.output)
            # TODO: figure out the function of get_updates_for
            self.update_inputs += play.model.get_updates_for(play.input)
            self.params_list += play.trainable_weights

        if self._unittest is True:
            self._x_feed_dict = { self.feed_inputs[0].name : _inputs.reshape(self.batch_input_shape) }

        self._y = [self.feed_mu, self.feed_sigma]

        assert len(self.feed_inputs) == 1, colors.red("ERROR: only need one input layer")
        ##################### Average outputs #############################
        if self._nb_plays > 1:
            self.y_pred = tf.keras.layers.Average()(self.model_outputs)
        else:
            self.y_pred = self.model_outputs[0]

        y_pred = tf.reshape(self.y_pred, shape=(-1,))
        self.diff = tf.math.subtract(y_pred[1:], y_pred[:-1])

        self.diff = tf.concat([tf.reshape(y_pred[0], shape=(1,)), self.diff], axis=0)
        self.curr_mu = tf.keras.backend.mean(self.diff)
        self.curr_sigma = tf.keras.backend.std(self.diff)

        with tf.name_scope('training'):
            if self._unittest is False:
                ###################### Calculate J by hand ###############################
                # 1. gradient for each play, assign to self.J_list_by_hand
                self.J_list_by_hand = [
                    gradient_all_layers(play.operator_layer.output,
                                        play.nonlinear_layer.output,
                                        play.operator_layer.kernel,
                                        play.nonlinear_layer.kernel,
                                        play.linear_layer.kernel,
                                        activation=self._activation) for play in self.plays]
            else:
                ###################### Calculate J by hand ###############################
                # 1. gradient for each play, assign to self.J_list_by_hand
                self.J_list_by_hand = [
                    gradient_all_layers(play.operator_layer.output,
                                        play.nonlinear_layer.output,
                                        play.operator_layer.kernel,
                                        play.nonlinear_layer.kernel,
                                        play.linear_layer.kernel,
                                        activation=self._activation,
                                        debug=True,
                                        inputs=self.feed_inputs[0],
                                        feed_dict=copy.deepcopy(self._x_feed_dict)) for play in self.plays]
                ####################### Calculate J by Tensorflow ###############################
                # 1. gradient for each play, assign to self.J_list_by_tf
                # 2. gradient for summuation of all plays, assign to self.J_by_tf
                model_outputs = [tf.reshape(self.model_outputs[i], shape=self.batch_input_shape) for i in range(self._nb_plays)]
                J_list_by_tf = [tf.keras.backend.gradients(model_output, self.feed_inputs) for model_output in model_outputs]
                self.J_list_by_tf = [tf.reshape(J_list_by_tf[i], shape=(1, -1, 1)) for i in range(self._nb_plays)]

                y_pred = tf.reshape(self.y_pred, shape=self.batch_input_shape)
                J_by_tf = tf.keras.backend.gradients(y_pred, self.feed_inputs)[0]
                self.J_by_tf = tf.reshape(J_by_tf, shape=(1, -1, 1))
            # (T * 1)
            # by hand
            J_by_hand = tf.reduce_mean(tf.concat(self.J_list_by_hand, axis=-1), axis=-1, keepdims=True)
            self.J_by_hand = tf.reshape(J_by_hand, shape=(-1,))

            if self._unittest is True:
                # we don't care about the graidents of loss function, it's calculated by tensorflow.
                return

            normalized_J_by_hand = tf.clip_by_value(tf.abs(self.J_by_hand), clip_value_min=1e-18, clip_value_max=1e18)

            # TODO: support derivation for p0
            # TODO: make loss customize from outside
            # TODO: learn mu/sigma/weights
            # NOTE: current version is fixing mu/sigma and learn weights
            # self._learnable_mu = True
            if self._learnable_mu:
                LOG.debug(colors.red("Using Learnable Mu Version"))
                # import ipdb;ipdb.set_trace()
                # from tensorflow.python.ops import variables as tf_variables
                # from tensorflow.keras.engine import base_layer_utils
                # self.mu = base_layer_utils.make_variable(name='mymodel/mu:0',
                #                                          initializer=tf.keras.initializers.Constant(value=0.0, dtype=tf.float32),
                #                                          trainable=True,
                #                                          dtype=tf.float32)
                # self.mu = tf.VariableV1(0.0, dtype=tf.float32, name='mymodel/mu')
                self._set_dtype_and_policy(dtype=tf.float32)

                self.mu = self.add_weight(
                    "mu",
                    dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(value=10.0, dtype=tf.float32),
                    trainable=True)
                self._initial_states_list = []
                for i in range(self._nb_plays):
                    self._initial_states_list.append(self.plays[i].operator_layer.states[0])
                    # self._initial_states_list.append(self.add_weight(
                    #     'learnable_state{}'.format(i),
                    #     dtype=tf.float32,
                    #     initializer=tf.keras.initializers.Constant(value=10.0, dtype=tf.float32),
                    #     trainable=True,
                    #     shape=(1, 1)
                    # ))

                self.loss_a = tf.keras.backend.square((self.diff - self.mu)/sigma) / 2
                self.params_list.append(self.mu)
                # self.params_list += self._initial_states_list

            else:
                self.loss_a = tf.keras.backend.square((self.diff - mu)/sigma) / 2
            self.loss_b = - tf.keras.backend.log(normalized_J_by_hand)
            # self.loss_by_hand = tf.keras.backend.mean(self.loss_a + self.loss_b)
            # self.loss_by_hand = tf.keras.backend.sum(self.loss_a + self.loss_b)
            self.loss_by_hand = tf.math.reduce_sum(self.loss_a + self.loss_b)
            # import ipdb; ipdb.set_trace()
            self.loss = self.loss_by_hand

            # 10-14
            self.reg_lambda = 0.001
            self.reg_mu = 0.001
            # 15-19
            # self.reg_lambda = 0.0001
            # self.reg_mu = 0.0001

            regularizers1 = [self.reg_lambda * tf.math.reduce_sum(tf.math.square(play.nonlinear_layer.kernel)) for play in self.plays]
            regularizers2 = [self.reg_mu * tf.math.reduce_sum(tf.math.square(play.linear_layer.kernel)) for play in self.plays]
            for regularizer in regularizers1:
                self.loss += regularizer

            for regularizer in regularizers2:
                self.loss += regularizer

            if outputs is not None:
                mse_loss1 = tf.keras.backend.mean(tf.square(self.y_pred - tf.reshape(outputs, shape=self.y_pred.shape)))
                mse_loss2 = tf.keras.backend.mean(tf.square(self.y_pred + tf.reshape(outputs, shape=self.y_pred.shape)))
            else:
                mse_loss1 = tf.constant(-1.0, dtype=tf.float32)
                mse_loss2 = tf.constant(-1.0, dtype=tf.float32)

            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=self.params_list,
                                                     loss=self.loss)
                # TODO: may be useful to uncomment the following code
                # updates += self.update_inputs
            training_inputs = self.feed_inputs + self.feed_targets

            import ipdb; ipdb.set_trace()
            # self.updates = updates
            # self.training_inputs = training_inputs
            if kwargs.get('test_stateful', False):
                outputs = [play.operator_layer.output for play in self.plays]
            elif self._learnable_mu:
                # outputs = [self.loss, mse_loss1, mse_loss2, self.diff, self.curr_sigma, self.curr_mu, self.y_pred, tf.keras.backend.mean(self.loss_a), tf.keras.backend.mean(self.loss_b), self.mu] + [play.operator_layer.output for play in self.plays]
                outputs = [self.loss, mse_loss1, mse_loss2, self.diff, self.curr_sigma, self.curr_mu, self.y_pred, tf.keras.backend.mean(self.loss_a), tf.keras.backend.mean(self.loss_b), self.mu] + [play.operator_layer.states[0] for play in self.plays] + self._initial_states_list
            else:
                outputs = [self.loss, mse_loss1, mse_loss2, self.diff, self.curr_sigma, self.curr_mu, self.y_pred, tf.keras.backend.mean(self.loss_a), tf.keras.backend.mean(self.loss_b)] + [play.operator_layer.output for play in self.plays]
            self.train_function = tf.keras.backend.function(training_inputs,
                                                            outputs,
                                                            updates=updates)

    def fit2(self,
             inputs,
             mu,
             sigma,
             outputs=None,
             epochs=100,
             verbose=0,
             steps_per_epoch=1,
             loss_file_name="./tmp/mymodel_loss_history.csv",
             learning_rate=0.001,
             decay=0.,
             preload_weights=False,
             weights_fname=None,
             **kwargs):

        writer = utils.get_tf_summary_writer("./log/mle")

        if outputs is not None:
            # glob ground-truth mu and sigma of outputs
            __mu__ = (outputs[1:] - outputs[:-1]).mean()
            __sigma__ = (outputs[1:] - outputs[:-1]).std()
            outputs = ops.convert_to_tensor(outputs, tf.float32)

        # self.compile(inputs, mu=mu, sigma=sigma, outputs=outputs, **kwargs)
        training_inputs, validate_inputs = inputs[:self._input_dim], inputs[self._input_dim:]
        if outputs is not None:
            training_outputs, validate_outputs = outputs[:self._input_dim], outputs[self._input_dim:]
        else:
            training_outputs, validate_outputs = None, None

        # kwargs['validate_inputs'] = validate_inputs
        # kwargs['validate_outputs'] = validate_outputs
        self.compile(training_inputs, mu=mu, sigma=sigma, outputs=training_outputs, **kwargs)
        # ins = self._x + self._y
        # ins = self._x
        input_dim =  self.batch_input_shape[-1]
        # ins = inputs.reshape(-1, 1, input_dim)
        utils.init_tf_variables()

        writer.add_graph(tf.get_default_graph())
        self.cost_history = []
        cost = np.inf
        patience_list = []
        prev_cost = np.inf

        # load weights pre-trained
        if preload_weights is True and weights_fname is not None:
            # find the best match weights from weights directory
            self.load_weights(weights_fname)

        if kwargs.get('test_stateful', False):
            outputs_list = [[] for _ in range(self._nb_plays)]
            assert epochs == 1, 'only epochs == 1 in unittest'
            for i in range(epochs):
                self.reset_states()
                for j in range(steps_per_epoch):
                    ins = inputs[j*input_dim:(j+1)*input_dim]
                    output = self.train_function([ins.reshape(1, 1, -1)])
                    states_list = [o.reshape(-1)[-1] for o in output]
                    self.reset_states(states_list=states_list)
                    for k, o in enumerate(output):
                        outputs_list[k].append(o.reshape(-1))

            results = []
            for output in outputs_list:
                results.append(np.hstack(output))
            assert len(results) == self._nb_plays
            return results

        if self._learnable_mu:
            logger_string_epoch = "Epoch: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}, loss_a: {:.7f}, loss_b: {:.7f}, learned_mu: {:.7f}"
        else:
            logger_string_epoch = "Epoch: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}, loss_a: {:.7f}, loss_b: {:.7f}"
        logger_string_step = "Steps: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}, loss_a: {:.7f}, loss_b: {:.7f}"

        _states_list = [1.0] * self._nb_plays

        for i in range(epochs):
            self.reset_states(_states_list)
            # self.reset_states()
            for j in range(steps_per_epoch):
                ins = inputs[j*input_dim:(j+1)*input_dim]
                if self._learnable_mu:
                    cost, mse_cost1, mse_cost2, diff_res, sigma_res, mu_res, y_pred, loss_a, loss_b, learned_mu, *operator_outputs = self.train_function([ins.reshape(1, 1, -1)])
                else:
                    cost, mse_cost1, mse_cost2, diff_res, sigma_res, mu_res, y_pred, loss_a, loss_b, *operator_outputs = self.train_function([ins.reshape(1, 1, -1)])
                states_list = [o.reshape(-1)[-1] for o in operator_outputs]
                self.reset_states(states_list=states_list)

                if prev_cost <= cost:
                    patience_list.append(cost)
                else:
                    prev_cost = cost
                    patience_list = []
                loss_by_hand, loss_by_tf = 0, 0
            if self._learnable_mu:
                LOG.debug(logger_string_epoch.format(i, float(cost), float(mse_cost1), float(mse_cost2), float(diff_res.mean()), float(diff_res.std()), float(__mu__), float(__sigma__), float(loss_by_hand), float(loss_by_tf), loss_a, loss_b, float(learned_mu)))
            else:
                LOG.debug(logger_string_epoch.format(i, float(cost), float(mse_cost1), float(mse_cost2), float(diff_res.mean()), float(diff_res.std()), float(__mu__), float(__sigma__), float(loss_by_hand), float(loss_by_tf), loss_a, loss_b))

            LOG.debug("Play States: {}".format(operator_outputs))
            LOG.debug("================================================================================")
            # save weights every 1000 epochs
            # if i % 1000 == 0 and i != 0:
            #     self.save_weights("{}-epochs-{}.h5".format(weights_fname[:-3], i))
            # import ipdb; ipdb.set_trace()
            # _states_list = operator_outputs[self._nb_plays:]
            self.cost_history.append([i, cost, mse_cost1, mse_cost2, loss_a, loss_b])

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1:], loss_file_name)

    def predict(self, inputs, individual=False):
        if isinstance(inputs, tf.Tensor):
            _inputs = inputs
        else:
            _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)

        _ = [play.build(_inputs) for play in self.plays if play.built is False]
        outputs = [play.predict(inputs) for play in self.plays]

        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        if individual is True:
            return prediction, outputs_
        return prediction

    def trend(self, prices, B, mu, sigma,
              start_pos=1000, end_pos=1100,
              delta=0.001, max_iteration=10000):
        # start_pos = 1000
        # # end_pos = 1100
        # end_pos = 1100

        # start_pos = 1000
        # end_pos = 1100
        # end_pos = 1010

        start_pos = 10
        # end_pos = 15
        end_pos = 110

        assert start_pos >= 0, colors.red("start_pos must be larger than 0")
        assert start_pos < end_pos, colors.red("start_pos must be less than end_pos")
        assert len(prices.shape) == 1, colors.red("Prices should be a vector")

        if not hasattr(self, '_batch_input_shape'):
            if hasattr(self.plays[0], '_batch_input_shape'):
                input_dim = self._batch_input_shape.as_list()[-1]
            else:
                raise Exception(colors.red("Not found batch_input_shape"))
        elif isinstance(self._batch_input_shape, (tf.TensorShape, )):
            input_dim = self._batch_input_shape[-1].value
        else:
            raise Exception(colors.red("Unknown **input_dim** error occurs in trend"))

        prices = np.hstack([prices[1500:2000],  prices[0:1000]])

        timestep = prices.shape[0] // input_dim
        shape = (1, timestep, input_dim)
        ################################################################################
        #                  Re-play the noise                                           #
        # original_prediction:                                                         #
        #   - expect the same size of prices                                           #
        # mu: use empirical mean                                                       #
        # sigma: use empirical standard derviation                                     #
        ################################################################################
        original_prediction = self.predict_parallel(prices)
        prices = prices[:original_prediction.shape[-1]]
        real_mu, real_sigma = mu, sigma
        if start_pos > 0:
            mu = (original_prediction[1:start_pos] - original_prediction[:start_pos-1]).mean()
            sigma = (original_prediction[1:start_pos] - original_prediction[:start_pos-1]).std()
        mu = 0
        sigma = 110

        LOG.debug(colors.cyan("emprical mean: {}, emprical standard dervation: {}".format(mu, sigma)))
        ################################################################################
        #                Decide the sign of predicted trends                           #
        ################################################################################
        counts = (((prices[1:] - prices[:-1]) >= 0) == ((original_prediction[1:]-original_prediction[:-1]) >= 0))
        sign = None

        # import ipdb; ipdb.set_trace()
        LOG.debug("(counts.sum() / prices.shape[0]) is: {}".format(counts.sum() / prices.shape[0]))
        if (counts.sum() / prices.shape[0]) <= 0.3:
            sign = +1
        elif (counts.sum() / prices.shape[0]) >= 0.7:
            sign = -1
        else:
            raise Exception(colors.red("The neural network doesn't train well"))

        # Enforce prediction to make sense
        original_prediction = original_prediction*sign
        ################################################################################
        #  My Pool
        ################################################################################
        start = time.time()
        for play in self.plays:
            task = Task(play, 'weights', None)
            self.pool.put(task)
        self.pool.join()
        weights_ = self.pool.results
        weights = [[], [], [], [], []]
        for w in weights_:
            weights[0].append(w[0])
            weights[1].append(w[1])
            weights[2].append(w[2])
            weights[3].append(w[3])
            weights[4].append(w[4])

        end = time.time()

        LOG.debug("Time cost during extract weights: {}".format(end-start))
        start = time.time()
        for play in self.plays:
            task = Task(play, 'operator_output', (prices,))
            self.pool.put(task)
        self.pool.join()

        operator_outputs = self.pool.results
        end = time.time()
        LOG.debug("Time cost during extract operator_outputs: {}".format(end-start))

        guess_prices = []
        k = start_pos
        seq = 1
        repeating = 100
        # repeating = 2

        nb_plays = self._nb_plays
        activation = self._activation
        start = time.time()
        pool = MP_CONTEXT.Pool(constants.CPU_COUNTS)
        # pool = MP_CONTEXT.Pool(1)
        args_list = []
        while k + seq - 1 < end_pos:
            prev_gt_price = prices[k-1]
            curr_gt_price = prices[k]
            prev_gt_prediction = original_prediction[k-1]
            curr_gt_prediction = original_prediction[k]
            args = (k,
                    seq,
                    repeating,
                    prev_gt_price,
                    curr_gt_price,
                    prev_gt_prediction,
                    curr_gt_prediction,
                    mu,
                    sigma,
                    nb_plays,
                    activation,
                    sign,
                    operator_outputs,
                    weights,
                    self._ensemble,
                    real_mu,
                    real_sigma)
            args_list.append(args)
            k += 1

        guess_prices = pool.map(wrapper_repeat, args_list)
        pool.close()
        pool.join()
        end = time.time()

        LOG.debug("Time cost for prediction price: {} s".format(end-start))

        LOG.debug("Verifing...")
        # import ipdb; ipdb.set_trace()
        guess_prices = np.array(guess_prices).reshape(-1)

        loss1 =  ((guess_prices - prices[start_pos:end_pos]) ** 2)
        loss2 = np.abs(guess_prices - prices[start_pos:end_pos])
        loss3 = (prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1]) ** 2
        loss4 = np.abs(prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1])

        LOG.debug("hnn-RMSE: {}".format((loss1.sum()/(end_pos-start_pos))**(0.5)))
        LOG.debug("baseline-RMSE: {}".format((loss3.sum()/(end_pos-start_pos))**(0.5)))
        LOG.debug("hnn-L1-ERROR: {}".format((loss2.sum()/(end_pos-start_pos))))
        LOG.debug("baseline-L1-ERROR: {}".format((loss4.sum()/(end_pos-start_pos))))

        return guess_prices

    def visualize_activated_plays(self, inputs, mu=0, sigma=1):
        input_dim = self._batch_input_shape[-1]
        points = inputs.shape[-1]
        timestamp = 1
        shape = (1, timestamp, input_dim)
        _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)
        _inputs = tf.reshape(_inputs, shape=shape)
        _outputs = [play.operator_layer(_inputs) for play in self.plays]
        outputs = utils.get_session().run(_outputs)
        outputs = [output.reshape(-1) for output in outputs]  # self._nb_plays * intputs.shape(-1)
        outputs = np.array(outputs)
        assert outputs.shape == (self._nb_plays, points)

        import seaborn as sns

        fig, ax = plt.subplots(figsize=(20, 20))
        x_size, y_size = 2, 1
        vmin, vmax = outputs.min(), outputs.max()
        assert x_size * y_size == self._nb_plays

        fig.set_tight_layout(True)
        x = np.linspace(0, x_size+1, x_size+2)
        y = np.linspace(0, y_size+1, y_size+2)
        xv, yv = np.meshgrid(x, y)
        fargs = (outputs, x_size, y_size, vmin, vmax, ax)
        global once
        once = True

        def update(i, *fargs):
            global once
            outputs = fargs[0]
            x_size = fargs[1]
            y_size = fargs[2]
            vmin = fargs[3]
            vmax = fargs[4]
            ax = fargs[5]
            LOG.info("Update animation frame: {}".format(i))
            output = outputs[:, i]
            output = output.reshape(x_size, y_size)
            sns.heatmap(output, linewidth=0.5, vmin=vmin, vmax=vmax, ax=ax, cbar=once)
            once = False

        anim = FuncAnimation(fig, update, frames=np.arange(0, points, 2),
                             fargs=fargs, interval=500)

        fname = './visualize-mu-{}-sigma-{}/heatmap1.gif'.format(mu, sigma)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        anim.save(fname, dpi=40, writer='imagemagick')


        fig, ax = plt.subplots(figsize=(20, 20))
        x_size, y_size = points // 20, 20
        vmin, vmax = outputs.min(), outputs.max()
        assert x_size * y_size == input_dim

        fig.set_tight_layout(True)
        x = np.linspace(0, x_size+1, x_size+2)
        y = np.linspace(0, y_size+1, y_size+2)
        xv, yv = np.meshgrid(x, y)
        fargs = (outputs, x_size, y_size, vmin, vmax, ax)

        once = True
        def update2(i, *fargs):
            global once
            outputs = fargs[0]
            x_size = fargs[1]
            y_size = fargs[2]
            vmin = fargs[3]
            vmax = fargs[4]
            ax = fargs[5]

            LOG.info("Update animation frame: {}".format(i))
            output = outputs[i, :]
            output = output.reshape(x_size, y_size)
            sns.heatmap(output, linewidth=0.5, vmin=vmin, vmax=vmax, ax=ax, cbar=once)
            once = False

        anim = FuncAnimation(fig, update2, frames=np.arange(0, self._nb_plays, 1),
                             fargs=fargs, interval=500)

        fname = './visualize-mu-{}-sigma-{}/heatmap2.gif'.format(mu, sigma)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        anim.save(fname, dpi=40, writer='imagemagick')


    def save_weights(self, fname):
        suffix = fname.split(".")[-1]
        assert suffix == 'h5', "must store in h5 format"

        for play in self.plays:
            path = "{}/{}plays/{}.{}".format(fname[:-3], len(self.plays), play._name, suffix)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()
            LOG.debug(colors.cyan("Saving {}'s Weights to {}".format(play._name, path)))
            play.save_weights(path)

        LOG.debug(colors.cyan("Writing input shape into disk..."))
        start = time.time()
        # import ipdb; ipdb.set_trace()

        with open("{}/{}plays/input_shape.txt".format(fname[:-3], len(self.plays)), "w") as f:
            f.write(":".join(map(str, self.batch_input_shape)))
        end = time.time()
        LOG.debug("Time cost during writing shape: {} s".format(end-start))

    def load_weights(self, fname, extra={}):
        LOG.debug(colors.cyan("Trying to Load Weights first..."))
        suffix = fname.split(".")[-1]
        dirname = "{}/{}plays".format(fname[:-3], self._nb_plays)
        if extra.get('use_epochs', False) is True or not os.path.isdir(dirname):
            LOG.debug(colors.red("Fail to Load Weights."))
            epochs = []
            base = '/'.join(fname.split('/')[:-1])
            for _dir in os.listdir(base):
                if os.path.isdir('{}/{}'.format(base, _dir)):
                    try:
                        epochs.append(int(_dir.split('-')[-1]))
                    except ValueError:
                        pass

            if not epochs:
                return False
            best_epoch = max(epochs)
            if extra.get('best_epoch', None) is not None:
                best_epoch = extra.get('best_epoch')

            LOG.debug("Loading weights from Epoch: {}".format(epochs))
            dirname = '{}-epochs-{}/{}plays'.format(fname[:-3], best_epoch, self._nb_plays)
            LOG.debug("Loading weights from epochs file: {}".format(dirname))
            if not os.path.isdir(dirname):
                # sanity checking
                raise Exception("Bugs inside *load_wegihts*")

        LOG.debug(colors.red("Found trained Weights. Loading..."))
        with open("{}/input_shape.txt".format(dirname), "r") as f:
            line = f.read()

        shape = list(map(int, line.split(":")))
        if 'shape' in extra:
            shape = extra['shape']

        self._batch_input_shape = tf.TensorShape(shape)

        if getattr(self, 'parallel_prediction'):
            for play in self.plays:
                play._batch_input_shape = tf.TensorShape(shape)
                play._preload_weights = False
                path = "{}/{}.{}".format(dirname, play._name, suffix)
                play._weights_fname = path
        else:
            start = time.time()
            for play in self.plays:
                if not play.built:
                    play._batch_input_shape = tf.TensorShape(shape)
                    play._preload_weights = True
                    play.build()
                path = "{}/{}.{}".format(dirname, play._name, suffix)
                play.load_weights(path)
                LOG.debug(colors.red("Set Weights for {}".format(play._name)))
            end = time.time()
            LOG.debug("Load weights cost: {} s".format(end-start))
        return True

    @property
    def trainable_weights(self):
        weights = []
        for play in self.plays:
            weights.append(play.trainable_weights)

        results = utils.get_session().run(weights)
        for i, result in enumerate(results):
            for j, r in enumerate(result):
                results[i][j] = results[i][j].reshape(-1)

        return results

    def __del__(self):
        LOG.debug("Start to close ProcessPool before deleting object")
        if hasattr(self, 'pool'):
            self.pool.close()

        tf.keras.backend.clear_session()
        utils.get_cache().clear()

    @property
    def states(self):
        # NOTE: doesn't work properly
        return [play.operator_layer.states for play in self.plays]

    def reset_states(self, states_list=None):
        if states_list is None:
            for i, play in enumerate(self.plays):
                play.operator_layer.reset_states()
            return

        if not isinstance(states_list[0], np.ndarray):
            states_list = [np.array([s]).reshape(1, 1) for s in states_list]
        else:
            states_list = [s.reshape(1, 1) for s in states_list]
        for i, play in enumerate(self.plays):
            play.operator_layer.reset_states(states_list[i])

    def reset_states_parallel(self, states_list=None):
        if states_list is None:
            states_list = [np.array([0]).reshape(1, 1) for _ in range(self._nb_plays)]
        elif not isinstance(states_list[0], np.ndarray):
            states_list = [np.array([s]).reshape(1, 1) for s in states_list]
        else:
            states_list = [s.reshape(1, 1) for s in states_list]
        for i, play in enumerate(self.plays):
            task = Task(play, 'reset_states', (states_list[i],))
            self.pool.put(task)
        self.pool.join()

    def get_op_outputs_parallel(self, inputs):
        for play in self.plays:
             task = Task(play, 'operator_output', (inputs,))
             self.pool.put(task)
        self.pool.join()
        return self.pool.results

    @property
    def batch_input_shape(self):
        '''
        return a list
        '''
        return self._batch_input_shape.as_list()

    def _load_sim_dataset(self, i):
        brief_data = np.loadtxt(self._fmt_brief.format(i), delimiter=',')
        truth_data = np.loadtxt(self._fmt_truth.format(i), delimiter=',')
        fake_data = np.loadtxt(self._fmt_fake.format(i), delimiter=',')

        fake_B1, fake_B2, fake_B3, _B1, _B2, _B3 = brief_data[0], brief_data[1], brief_data[2], brief_data[3], brief_data[4], brief_data[5]
        fake_price_list, fake_stock_list = fake_data[:, 0], fake_data[:, 1]
        price_list, stock_list = truth_data[:, 0], truth_data[:, 1]
        return fake_price_list, fake_stock_list, price_list, stock_list, fake_B1, fake_B2, fake_B3, _B1, _B2, _B3

    def plot_graphs_together(self, prices, noises, mu, sigma, ensemble_mode=False):
        self._fmt_brief = '../simulation/training-dataset/mu-0-sigma-110.0-points-2000/{}-brief.csv'
        self._fmt_truth = '../simulation/training-dataset/mu-0-sigma-110.0-points-2000/{}-true-detail.csv'
        self._fmt_fake = '../simulation/training-dataset/mu-0-sigma-110.0-points-2000/{}-fake-detail.csv'
        length = 1000
        assert length <= prices.shape[-1] - 1, "Length must be less than prices.shape-1"
        batch_size = self.batch_input_shape[-1]
        # states_list = None
        prices_tensor = tf.reshape(ops.convert_to_tensor(prices, dtype=tf.float32), shape=(1, 1, -1))

        results = self.predict_parallel(prices)
        self.reset_states_parallel(states_list=None)
        operator_outputs = self.get_op_outputs_parallel(prices)
        self.reset_states_parallel(states_list=None)
        prices = prices[:results.shape[-1]]
        # import ipdb; ipdb.set_trace()
        counts = ((prices[1:]-prices[:-1] >= 0) == (results[1:] - results[:-1] >= 0)).sum()
        sign = None
        if counts / prices.shape[0] >= 0.65:
            sign = -1
        elif counts / prices.shape[0] <= 0.35:
            sign = +1

        LOG.debug("The counts is: {}, percentage is: {}".format(counts, counts/prices.shape[0]))
        if sign is None:
            raise Exception("the neural network doesn't train well, counts is {}".format(counts))

        # determine correct direction of results
        states_list = None

        _result_list = []
        packed_results = []
        bifurcation_list = []

        for i in range(length):
             # fig, (ax1, ax2) = plt.subplots(2, sharex='all')

            fake_price_list, fake_noise_list, price_list, noise_list, fake_B1, fake_B2, fake_B3, _B1, _B2, _B3 = self._load_sim_dataset(i)
            start_price, end_price = price_list[0], price_list[-1]
            if abs(prices[i] - start_price) > 1e-7 or \
              abs(prices[i+1] - end_price) > 1e-7:
                LOG.error("Bugs: prices is out of expectation")

            interpolated_prices = np.linspace(start_price, end_price, batch_size)
            # self.reset_states_parallel(states_list=states_list)
            interpolated_noises = self.predict_parallel(interpolated_prices, states_list=states_list)
            # FOR DEBUG
            _result_list.append(interpolated_noises[-1])
            # result_list.append(interpolated_noises[0])

            fake_start_price, fake_end_price = fake_price_list[0], fake_price_list[-1]
            fake_interpolated_prices = np.linspace(fake_start_price, fake_end_price, batch_size)
            # self.reset_states_parallel(states_list=states_list)
            fake_interpolated_noises = self.predict_parallel(fake_interpolated_prices, states_list=states_list)

            # fake_interpolated_prices = interpolated_prices
            # fake_interpolated_noises = interpolated_noises
            # NOTE: correct here, don't change
            states_list = [o[i] for o in operator_outputs]

            fake_size = fake_price_list.shape[-1]
            if fake_size >= 50:
                if fake_size // 50 <= 1:
                    fake_size = 100
                fake_price_list_ = fake_price_list[::fake_size // 50]
                fake_price_list = np.hstack([fake_price_list_, fake_price_list[-1]])
                fake_noise_list_ = fake_noise_list[::fake_size // 50]
                fake_noise_list = np.hstack([fake_noise_list_, fake_noise_list[-1]])

            size = price_list.shape[-1]
            if size >= 50:
                if size // 50 <= 1:
                   size = 100
                price_list_ = price_list[::size // 50]
                price_list = np.hstack([price_list_, price_list[-1]])
                noise_list_ = noise_list[::size // 50]
                noise_list = np.hstack([noise_list_, noise_list[-1]])

            # fake_price_list = price_list
            # fake_noise_list = noise_list

            # price_list = fake_price_list
            # noise_list = fake_noise_list


            # import ipdb; ipdb.set_trace()
            fake_size = fake_price_list.shape[-1]
            fake_step = batch_size//fake_size
            if fake_step <= 0:
                fake_step = 1

            fake_interpolated_prices_ = fake_interpolated_prices[::fake_step]
            fake_interpolated_noises_ = fake_interpolated_noises[::fake_step]
            fake_interpolated_prices = np.hstack([fake_interpolated_prices_, fake_interpolated_prices[-1]])
            fake_interpolated_noises = np.hstack([fake_interpolated_noises_, fake_interpolated_noises[-1]])

            size = price_list.shape[-1]
            step = batch_size // size
            if step <= 0:
                step = 1
            interpolated_prices_ = interpolated_prices[::step]
            interpolated_noises_ = interpolated_noises[::step]
            interpolated_prices = np.hstack([interpolated_prices_, interpolated_prices[-1]])
            interpolated_noises = np.hstack([interpolated_noises_, interpolated_noises[-1]])

            # fake_interpolated_prices = interpolated_prices
            # fake_interpolated_noises = interpolated_noises

            # interpolated_prices = fake_interpolated_prices
            # interpolated_noises = fake_interpolated_noises


            interpolated_noises = sign * interpolated_noises
            fake_interpolated_noises = sign * fake_interpolated_noises

            if ensemble_mode is True:
                packed_results.append([fake_price_list, fake_noise_list, price_list, noise_list,
                                       fake_interpolated_prices, fake_interpolated_noises, interpolated_prices, interpolated_noises,
                                       fake_B1, fake_B2, fake_B3, _B1, _B2, _B3])
                continue

            fig, ax1 = plt.subplots(1, figsize=(10, 10))

            is_bifurcation_1, is_correct_direction_of_noise_1 = self._plot_sim(ax1, fake_price_list, fake_noise_list,
                                                                               price_list, noise_list, fake_B1,
                                                                               fake_B2, fake_B3, _B1, _B2, _B3)
            is_bifurcation_2, is_correct_direction_of_noise_2 = self._plot_interpolated(ax1, fake_interpolated_prices, fake_interpolated_noises,
                                                                                        interpolated_prices, interpolated_noises, fake_B1,
                                                                                        fake_B2, fake_B3, _B1, _B2, _B3)
            # is_bifurcation_1, is_correct_direction_of_noise_1 = self._detect_bifurcation(price_list, noise_list)
            # is_bifurcation_2, is_correct_direction_of_noise_2 = self._detect_bifurcation(interpolated_prices, interpolated_noises)
            bifurcation_list.append([is_bifurcation_1, is_bifurcation_2, is_correct_direction_of_noise_1, is_correct_direction_of_noise_2])
            # import ipdb; ipdb.set_trace()
            if mu is None and sigma is None:
                fname = './frames/{}.png'.format(i)
            else:
                fname = './frames-nb_plays-{}-units-{}-batch_size-{}-mu-{}-sigma-{}/ensemble-{}/{}.png'.format(
                    self._nb_plays, self._units, self._input_dim,
                    mu, sigma,
                    self._ensemble,
                    i)

            LOG.debug("plot {}".format(fname))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fig.savefig(fname, dpi=100)

        fname = './frames-nb_plays-{}-units-{}-batch_size-{}-mu-{}-sigma-{}/ensemble-{}/bifurcation.csv'.format(
            self._nb_plays, self._units, self._input_dim,
            mu, sigma,
            self._ensemble)
        # import ipdb; ipdb.set_trace()
        bifurcation_list = np.array(bifurcation_list).astype(int)

        np.savetxt(fname, bifurcation_list, fmt="%s", delimiter=',', header='#ground-truth-bifurcation,#neural-network-bifurcation,#ground-truth-direction,#neural-network-direction')

        return packed_results

        # result_list.append(interpolated_noises[-1])
        # result_list = np.array(result_list)
        # import ipdb; ipdb.set_trace()
        # if np.allclose(results, result_list) is True:
        #     print("Correct")
        # else:
        #     import ipdb; ipdb.set_trace()
        #     print("Hello world")
    def _detect_bifurcation(self, price_list, noise_list):
        # Detect bifurcation and predict correct noise ?
        if price_list[-1] - price_list[0] > 0:
            # price rises, find maximum value of noise
            h1 = abs(np.max(noise_list))
        else:
            # price decreases, find minimum value of noise
            h1 = abs(np.min(noise_list))

        h2 = np.max(noise_list) - np.min(noise_list)
        ratio = h1/h2

        flag = not ((price_list[-1] > price_list[0]) ^ (noise_list[-1] < noise_list[0]))

        return (ratio >= 0.1), flag


    def _plot_sim(self, ax,
                  fake_price_list, fake_noise_list,
                  price_list, noise_list,
                  fake_B1, fake_B2, fake_B3,
                  _B1, _B2, _B3, color='blue', plot_target_line=True):

        fake_l = 10 if len(fake_price_list) == 1 else len(fake_price_list)
        l = 10 if len(price_list) == 1 else len(price_list)
        fake_B1, fake_B2, fake_B3 = np.array([fake_B1]*fake_l), np.array([fake_B2]*fake_l), np.array([fake_B3]*fake_l)
        _B1, _B2, _B3 = np.array([_B1]*l), np.array([_B2]*l), np.array([_B3]*l)

        if plot_target_line is True:
            fake_B2 = fake_B2 - fake_B1
            fake_B3 = fake_B3 - fake_B1
            fake_noise_list = fake_noise_list - fake_B1
            fake_B1 = fake_B1 - fake_B1

            _B2 = _B2 - _B1
            _B3 = _B3 - _B1
            noise_list = noise_list - _B1
            _B1 = _B1 - _B1
            ax.plot(fake_price_list, fake_B1, 'r', fake_price_list, fake_B2, 'c--', fake_price_list, fake_B3, 'k--')
            ax.plot(price_list, _B1, 'r', price_list, _B2, 'c', price_list, _B3, 'k-')

        else:
            fake_noise_list = fake_noise_list - fake_B1
            fake_B1 = fake_B1 - fake_B1
            noise_list = noise_list - _B1
            _B1 = _B1 - _B1
            pass

        # import ipdb; ipdb.set_trace()
        ax.plot(fake_price_list, fake_noise_list, color=color, marker='s', markersize=3, linestyle='--')
        ax.plot(price_list, noise_list, color=color, marker='.', markersize=6, linestyle='-')
        ax.set_xlabel("Prices")
        ax.set_ylabel("#Noise")


        is_bifurcation, is_correct_direction_of_noise = self._detect_bifurcation(price_list, noise_list)
        if is_bifurcation:
            ax.text(1.1*price_list.mean(), 0.9*noise_list.mean(), "bifurcation", color=color)
        else:
            ax.text(0.9*price_list.mean(), 0.9*noise_list.mean(), "non-bifurcation", color=color)
        if is_correct_direction_of_noise:
            ax.text(price_list.mean(), noise_list.mean(), 'True', color=color)
        else:
            ax.text(price_list.mean(), noise_list.mean(), 'False', color=color)

        return is_bifurcation, is_correct_direction_of_noise

            # Detect bifurcation and predict correct noise ?
        # if price_list[-1] - price_list[0] > 0:
        #     # price rises, find maximum value of noise
        #     h1 = abs(np.max(noise_list))
        # else:
        #     h1 = abs(np.min(noise_list))
        # h2 = np.max(noise_list) - np.min(noise_list)
        # ratio = h1/h2

        # if ratio >= 0.1:
        #     ax.text(1.1*price_list.mean(), 0.9*noise_list.mean(), "bifurcation", color=color)
        #             # horizontalalignment='right',
        #             # verticalalignment='bottom',
        #             # transform=ax.transAxes)
        # else:
        #     ax.text(0.9*price_list.mean(), 0.9*noise_list.mean(), "non-bifurcation", color=color)
        #     # ax.text(0.75, 0.8, "non-bifurcation", color=color,
        #     #         horizontalalignment='right',
        #     #         verticalalignment='bottom',
        #     #         transform=ax.transAxes)
        # # import ipdb; ipdb.set_trace()
        # flag = not (price_list[-1] > price_list[0]) ^ (noise_list[-1] < noise_list[0])
        # if flag:
        #     ax.text(price_list.mean(), noise_list.mean(), 'True', color=color)
        # else:
        #     ax.text(price_list.mean(), noise_list.mean(), 'False', color=color)


    def _plot_interpolated(self, ax,
                           fake_interpolated_prices,
                           fake_interpolated_noises,
                           interpolated_prices,
                           interpolated_noises,
                           fake_B1, fake_B2, fake_B3,
                           _B1, _B2, _B3,
                           color=mcolors.CSS4_COLORS['orange']):

        return self._plot_sim(ax,
                              fake_interpolated_prices, fake_interpolated_noises,
                              interpolated_prices, interpolated_noises,
                              fake_interpolated_noises[0], fake_B2, fake_B3,
                              interpolated_noises[0], _B2, _B3, color, plot_target_line=False)


class EnsembleModel(object):

    def __init__(self,
                 ensembles,
                 input_dim,
                 timestep,
                 units,
                 activation,
                 nb_plays,
                 parallel_prediction,
                 best_epoch_list,
                 use_epochs=False):

        self._input_dim = input_dim
        self._timestep = timestep
        self._units = units
        self._activation = activation
        self._nb_plays = nb_plays
        self._batch_size = input_dim
        self._models = []
        self._ensembles = ensembles
        for ensemble in ensembles:
            self._models.append(
                MyModel(input_dim=input_dim,
                        timestep=timestep,
                        units=units,
                        activation=activation,
                        nb_plays=nb_plays,
                        parallel_prediction=parallel_prediction,
                        ensemble=ensemble)
            )
        self._shape = (1, timestep, input_dim)
        self._parallelism = parallel_prediction
        self._best_epoch_list = best_epoch_list
        self._use_epochs = use_epochs

    def load_weights(self):
        models_diff_weights_mc_stock_model_saved_weights = './new-dataset/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}weights-batch_size-{batch_size}.h5'
        for i, model in enumerate(self._models):
            weights_fname = models_diff_weights_mc_stock_model_saved_weights.format(
                method='sin',
                activation=None,
                state=0,
                mu=0,
                sigma=110,
                units=20,
                nb_plays=20,
                points=1000,
                input_dim=1,
                __activation__=self._activation,
                __units__=self._units,
                __mu__=0,
                __sigma__=110,
                __state__=0,
                __nb_plays__=self._nb_plays,
                loss='mle',
                ensemble=model._ensemble,
                batch_size=self._input_dim)
            model.load_weights(weights_fname,
                               extra={'shape': self._shape,
                                      'parallelism': self._parallelism,
                                      'best_epoch': self._best_epoch_list[i],
                                      'use_epochs': self._use_epochs})

    def plot_graphs_together(self, prices, noises, mu, sigma):
        packed_result_list = []
        for i, model in enumerate(self._models):
            res = model.plot_graphs_together(prices=prices,
                                             noises=noises,
                                             mu=mu,
                                             sigma=sigma,
                                             ensemble_mode=True)
            packed_result_list.append(res)

        N = len(res)
        M = len(self._models)
        new_packed_result_list = [[0 for _ in range(M)] for _ in range(N)]
        for i in range(M):
            for j in range(N):
                new_packed_result_list[j][i] = packed_result_list[i][j]

        colors = ['green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'gold', 'olive', 'grey', 'saddlebrown', 'purple', 'plum', 'pink', 'burlywood', 'darkkhaki', 'deepskyblue', 'turquoise', 'lime', 'thistle', 'mediumvioletred']

        for i, packed_result in enumerate(new_packed_result_list):
            fig, ax1 = plt.subplots(1, figsize=(10, 10))
            avg_fake_noise_list = avg_noise_list = avg_fake_interpolated_noise_list = avg_interpolated_noise_list = None

            for j, result in enumerate(packed_result):
                color = colors[self._models[j]._ensemble]
                fake_price_list, fake_noise_list, price_list, noise_list, fake_interpolated_prices, fake_interpolated_noises, interpolated_prices, interpolated_noises, fake_B1, fake_B2, fake_B3, _B1, _B2, _B3 = result
                if avg_fake_noise_list is None:
                    avg_fake_noise_list = fake_noise_list
                    avg_noise_list = noise_list
                    avg_fake_interpolated_noise_list = fake_interpolated_noises
                    avg_interpolated_noise_list = interpolated_noises
                else:
                    avg_fake_noise_list += fake_noise_list
                    avg_noise_list += noise_list
                    avg_fake_interpolated_noise_list += fake_interpolated_noises
                    avg_interpolated_noise_list += interpolated_noises

                self._models[0]._plot_sim(ax1, fake_price_list, fake_noise_list,
                                          price_list, noise_list, fake_B1,
                                          fake_B2, fake_B3, _B1, _B2, _B3)
                self._models[0]._plot_interpolated(ax1, fake_interpolated_prices, fake_interpolated_noises,
                                                   interpolated_prices, interpolated_noises, fake_B1,
                                                   fake_B2, fake_B3, _B1, _B2, _B3, color=color)

            avg_fake_noise_list /= M
            avg_noise_list /= M
            avg_fake_interpolated_noise_list /= M
            avg_interpolated_noise_list /= M

            self._models[0]._plot_sim(ax1, fake_price_list, avg_fake_noise_list,
                                      price_list, avg_noise_list, fake_B1,
                                      fake_B2, fake_B3, _B1, _B2, _B3)
            self._models[0]._plot_interpolated(ax1, fake_interpolated_prices, avg_fake_interpolated_noise_list,
                                               interpolated_prices, avg_interpolated_noise_list, fake_B1,
                                               fake_B2, fake_B3, _B1, _B2, _B3)

            fname = './frames-nb_plays-{}-units-{}-batch_size-{}-mu-{}-sigma-{}/ensemble/{}.png'.format(
                self._nb_plays, self._units, self._input_dim, mu, sigma, i)
            LOG.debug("plot {}".format(fname))

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fig.savefig(fname, dpi=100)

    def trend(self):
        import trading_data as tdata
        models_diff_weights_mc_stock_model_trends = './new-dataset/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}/trends-batch_size-{batch_size}.csv'
        # import ipdb; ipdb.set_trace()
        avg = []
        # for i, model in enumerate(self._models):
        for ensemble in self._ensembles:
            predicted_fname = models_diff_weights_mc_stock_model_trends.format(
                method='sin',
                activation=None,
                state=0,
                mu=0,
                sigma=110,
                units=20,
                nb_plays=20,
                points=1000,
                input_dim=1,
                __activation__=self._activation,
                __units__=self._units,
                __mu__=0,
                __sigma__=110,
                __state__=0,
                __nb_plays__=self._nb_plays,
                loss='mle',
                ensemble=ensemble,
                batch_size=1500)

            a, b = tdata.DatasetLoader.load_data(predicted_fname)
            avg.append(b)

        avg = np.vstack(avg).mean(axis=0)

        colors = ['green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'gold', 'olive', 'grey', 'saddlebrown', 'purple', 'plum', 'pink', 'burlywood', 'darkkhaki', 'deepskyblue', 'turquoise', 'lime', 'thistle', 'mediumvioletred']

        formatter = './new-dataset/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble/loss-{loss}/trends-batch_size-{batch_size}.csv'
        ensemble_trend_fname = formatter.format(
                method='sin',
                activation=None,
                state=0,
                mu=0,
                sigma=110,
                units=20,
                nb_plays=20,
                points=1000,
                input_dim=1,
                __activation__=self._activation,
                __units__=self._units,
                __mu__=0,
                __sigma__=110,
                __state__=0,
                __nb_plays__=self._nb_plays,
                loss='mle',
                batch_size=self._input_dim)

        tdata.DatasetSaver.save_data(a, avg, ensemble_trend_fname)
        LOG.debug("Generate ensemble results {}".format(ensemble_trend_fname))
