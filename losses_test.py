import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import losses
import utils
import log as logging
import core
import trading_data as tdata
import colors

LOG = logging.getLogger(__name__)
sess = utils.get_session()


if __name__ == "__main__":
    LOG.debug("********************{}********************".format(colors.red("run under debug mode")))
    LOG.debug("==================Test core.Phi===========================")
    # case 1.
    a = tf.constant(1.)
    b = core.Phi(a)
    g = tf.gradients(b, [a])
    LOG.debug("expected gradient: 1.0, evaluate gradient: {}".format(sess.run(g)))
    # case 2.
    a = tf.constant(-0.5)
    b = core.Phi(a)
    g = tf.gradients(b, [a])
    LOG.debug("expected gradient: 0.0, evaluate gradient: {}".format(sess.run(g)))
    # case 3.
    a = tf.constant(-2.)
    b = core.Phi(a)
    g = tf.gradients(b, [a])
    LOG.debug("expected gradient: 1.0, evaluate gradient: {}".format(sess.run(g)))
    LOG.debug("==========================================================")

    LOG.debug("==================Test core.PlayCell=========================")
    # playcell gradient
    points = 10
    inputs = [-0.5, 1, -2, 5]

    LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = sin(t) + 3 sin(1.3 t)  + 1.2 sin (1.6 t)]")))
    weight = float(2)
    width = float(1)
    state = float(-0.5)
    cell = core.PlayCell(weight=weight, width=width, debug=True)
    outputs = cell(inputs, state)
    g = tf.gradients(outputs, [cell._inputs, cell._state])
    LOG.debug(sess.run(g))
    LOG.debug("============================================================")

    # play gradient
    LOG.debug("====================Test core.Play==========================")
    layer = core.Play(units=4, cell=cell, debug=True)
    outputs = layer(inputs)
    g = tf.gradients(outputs, [layer._inputs])
    LOG.debug(inputs)
    LOG.debug(sess.run(outputs))
    LOG.debug(sess.run(g))
    LOG.debug("============================================================")

    # playmodel gradient
    LOG.debug("====================Test core.PlayModel==========================")
    nb_plays = 1
    inputs = np.array([-0.5, 1, -2, 5]).astype(np.float32)
    play_model = core.PlayModel(nb_plays=nb_plays, debug=True)
    outputs = play_model(inputs)
    g = tf.gradients(outputs, [play_model.inputs])
    # LOG.debug(sess.run(outputs))
    LOG.debug("gradient: {}".format(sess.run(g)))
    LOG.debug("============================================================")

    LOG.debug("====================Test core.PlayModel2==========================")
    nb_plays = 1
    play_model2 = core.PlayModel2(nb_plays=nb_plays, debug=True)
    inputs = np.array([-0.5, 1, -2, 5]).astype(np.float32)
    outputs = play_model2(inputs)
    g = tf.gradients(outputs, [play_model2.inputs])

    LOG.debug(sess.run(g))
    LOG.debug("============================================================")

    LOG.debug("********************{}********************".format(colors.red("run without debug mode")))
    points = 10
    inputs = [-0.5, 1, -2, 5]

    LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = sin(t) + 3 sin(1.3 t)  + 1.2 sin (1.6 t)]")))
    weight = float(2)
    width = float(1)
    state = float(-0.5)
    cell = core.PlayCell(weight=weight, width=width)
    outputs = cell(inputs, state)

    # initalized all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    g = tf.gradients(outputs, [cell._inputs, cell._state])
    LOG.debug(sess.run(g))
    LOG.debug("============================================================")

    # play gradient
    LOG.debug("====================Test core.Play==========================")
    layer = core.Play(units=4, cell=cell)
    outputs = layer(inputs)

    # initalized all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    g = tf.gradients(outputs, [layer._inputs])
    LOG.debug(inputs)
    LOG.debug(sess.run(outputs))
    LOG.debug(sess.run(g))
    LOG.debug("============================================================")

    # playmodel gradient
    LOG.debug("====================Test core.PlayModel2==========================")
    nb_plays = 1
    play_model2 = core.PlayModel2(nb_plays=nb_plays, debug=True)
    inputs = np.array([-0.5, 1, -2, 5]).astype(np.float32)
    outputs = play_model2(inputs)

    # initalized all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    g = tf.gradients(outputs, [play_model2._inputs])
    LOG.debug(sess.run(g))
    LOG.debug("============================================================")
