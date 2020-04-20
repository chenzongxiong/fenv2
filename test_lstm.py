import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import utils
import constants
import trading_data as tdata
import colors
import log as logging
from core import confusion_matrix


LOG = logging.getLogger(__name__)


def lstm_regression(x,
                    y,
                    units=1,
                    epochs=1000,
                    optimizer='adam'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mse', optimizer=optimizer)

    train_x, test_x = x[:1000].reshape((1, -1, 1)), x[1000:].reshape((1, -1, 1))
    train_y, test_y = y[:1000].reshape((1, -1, 1)), y[1000:]
    train_x_tensor = ops.convert_to_tensor(train_x, dtype=tf.float32)
    train_y_tensor = ops.convert_to_tensor(train_y, dtype=tf.float32)

    steps_per_epoch = 1

    model.fit(train_x_tensor, train_y_tensor, verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    results = model.evaluate(train_x_tensor, train_y_tensor, steps=steps_per_epoch)
    print("Results: {}".format(results))
    test_x_tensor = ops.convert_to_tensor(test_x, dtype=tf.float32)
    predict_y = model.predict(test_x_tensor, steps=steps_per_epoch)
    predict_y = predict_y.reshape(-1)
    y = np.vstack([test_y[:1000], predict_y]).T

    return test_x.reshape(-1)[:1000], y


if __name__ == "__main__":
    method = 'sin'
    mu = 0
    sigma = 110
    points = 1000
    state = 0
    activation = None
    input_dim = 1
    nb_plays = 20
    __units__ = 64

    input_file_key = 'models_diff_weights_mc_stock_model'
    predict_file_key = 'models_diff_weights_mc_stock_model_predictions'
    fname = constants.DATASET_PATH[input_file_key].format(method=method,
                                                          activation=activation,
                                                          state=state,
                                                          mu=mu,
                                                          sigma=sigma,
                                                          points=points,
                                                          input_dim=input_dim,
                                                          nb_plays=nb_plays)

    predict_fname = constants.DATASET_PATH[predict_file_key].format(method=method,
                                                                    activation=activation,
                                                                    state=state,
                                                                    mu=mu,
                                                                    sigma=sigma,
                                                                    points=points,
                                                                    input_dim=input_dim,
                                                                    nb_plays=nb_plays,
                                                                    units=10000,
                                                                    __activation__='tanh',
                                                                    __state__=0,
                                                                    __units__=__units__,
                                                                    __nb_plays__=0,
                                                                    loss='mse')

    prices, noise = tdata.DatasetLoader.load_data(fname)
    timesteps = np.arange(prices.shape[-1])
    input_x, y = lstm_regression(timesteps, prices, units=__units__)
    tdata.DatasetSaver.save_data(input_x, y, predict_fname)

    a, b = tdata.DatasetLoader.load_data(predict_fname)
    confusion = confusion_matrix(b[:, 0], b[:, 1])
    LOG.debug(colors.purple("confusion matrix is: {}".format(confusion)))
