import log as logging
import tensorflow as tf



LOG = logging.getLogger(__name__)



def jacobian(fx, x, parallel_iterations=10):
    '''
    https://stackoverflow.com/questions/48878053/tensorflow-gradient-with-respect-to-matrix
    Given a tensor fx, which is a function of x, vectorize fx (via tf.reshape(fx, [-1])),
    and then compute the jacobian of each entry of fx with respect to x.
    Specifically, if x has shape (m,n,...,p), and fx has L entries (tf.size(fx)=L), then
    the output will be (L,m,n,...,p), where output[i] will be (m,n,...,p), with each entry denoting the
    gradient of output[i] wrt the corresponding element of x.
    '''
    # return map(lambda fxi: tf.gradients(fxi, x)[0],
    #            tf.reshape(fx, [-1]),
    #            dtype=x.dtype,
    #            parallel_iterations=parallel_iterations)


def mle_loss(B, W, mu, tau):
    """
    Calculate maximize log-likelihood
    Parameters:
    --------
    B: N by 1 vector. dtype: tf.float32
    W: N by N matrix (?). dtype: tf.float32
    mu: a scalar. dtype: tf.float32
    tau: a scalar, same as sigma in gaussian distribution. dtype: tf.float32
    """
    J = tf.gradient(B, [W])
    _B = tf.math.squre(B[1:] - B[:-1] - mu) - tf.math.log(tau)
    # TODO: multiply by Jaccobia matrix, unclear
    log_prob = _B * tf.linalg.det(J)

    neg_log_likelihood = tf.reduce_sum(log_prob)

    return neg_log_likelihood


def mle_gradient(loss, W, mu, tau, P0):
    return tf.gradient(loss, [W, mu, tau, P0])


def linear(x, width=1.0):
    return x


if __name__ == "__main__":
    import core
    import utils
    import numpy as np
    inputs = [-0.5, 1, -2, 5]
    inputs = np.array(inputs).astype(np.float32)
    inputs.reshape(1, -1)
    state = float(2)

    cell = core.PlayCell(hysteretic_func=linear, debug=True)
    outputs = cell(inputs, state)
    sess = utils.get_session()
    LOG.debug("inputs: {}".format(sess.run(cell._inputs)))
    LOG.debug("outputs: {}".format(sess.run(outputs)))

    for x in range(outputs.shape[0].value):
        J1 = tf.gradients(outputs[x], cell.kernel)
        LOG.debug("J1: {}".format(sess.run(J1)))

    # J1 = tf.gradients(outputs, [cell._inputs, cell.kernel])
    # outputs = tf.reshape(outputs, shape=(-1,))
    # J2 = tf.gradients(outputs[0], [cell.kernel])

    # LOG.debug("J2: {}".format(sess.run(J2)))
    # mu = 0.0
    # tau = 1.0
    # loss = mle_loss(outputs, cell.kernel, mu, tau)
