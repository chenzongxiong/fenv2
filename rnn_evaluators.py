import sys
import argparse

import pool

import constants
import log as logging

LOG = logging.getLogger(__name__)

methods = constants.METHODS
weights = constants.WEIGHTS
widths = constants.WIDTHS
nb_plays = constants.NB_PLAYS
epochs = constants.EPOCHS
epochs = 100


class Evaluator(object):

    @classmethod
    def simple_rnn(cls, units):
        import tensorflow as tf
        cls.optimizer = tf.train.GradientDescentOptimizer(0.01)
        cls.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.SimpleRNN(units, input_shape=(units, 1)))
        cls.model.compile(loss='mse', optimizer=cls.optimizer, metrics=['mse'])
        return cls

    @classmethod
    def lstm(cls, units):
        import tensorflow as tf
        cls.optimizer = tf.train.GradientDescentOptimizer(0.01)
        cls.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)

        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.LSTM(units, input_shape=(units, 1)))
        cls.model.compile(loss='mse', optimizer=cls.optimizer, metrics=['mse'])
        return cls

    @classmethod
    def gru(cls, units):
        import tensorflow as tf
        cls.optimizer = tf.train.GradientDescentOptimizer(0.01)
        cls.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)

        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.GRU(units, input_shape=(units, 1)))
        cls.model.compile(loss='mse', optimizer=cls.optimizer, metrics=['mse'])
        return cls

    @classmethod
    def fit(cls, inputs, outputs):
        cls.model.fit(inputs, outputs, epochs=epochs,
                      verbose=1, callbacks=[cls.early_stopping_callback])
        return cls

    @classmethod
    def evaluate(cls, inputs, outputs, fname=None):
        loss, mse = cls.model.evaluate(inputs, outputs, verbose=1)
        LOG.debug("loss: {}, mse: {}".format(loss, mse))
        if fname:
            import trading_data as tdata
            tdata.DatasetSaver.save_loss({"loss": loss, "mse": mse}, fname)
        return cls

    @classmethod
    def predict(cls, inputs, outputs, fname):
        predictions = cls.model.predict(inputs)
        if fname:
            import trading_data as tdata
            inputs = inputs.reshape(-1)
            predictions = predictions.reshape(-1)
            tdata.DatasetSaver.save_data(inputs, predictions, fname)
        return predictions


def loop(method, weight, width, nb_plays, neural_type="simple_rnn"):
    import tensorflow as tf
    import trading_data as tdata

    fname = constants.FNAME_FORMAT["models"].format(method=method, weight=weight,
                                                    width=width, nb_plays=nb_plays)
    inputs, outputs = tdata.DatasetLoader.load_data(fname)
    inputs, outputs = inputs[:1000], outputs[:1000]
    inputs = inputs.reshape(1, -1, 1)
    outputs = outputs.reshape(1, -1)
    units = outputs.shape[-1]

    LOG.debug("neural type is: {}, units is: {}".format(neural_type, units))
    if neural_type == "simple_rnn":
        rnn_loss_fname = constants.FNAME_FORMAT["models_rnn_loss"].format(method=method, weight=weight,
                                                                          width=width, nb_plays=nb_plays)
        rnn_predictions_fname = constants.FNAME_FORMAT["models_rnn_predictions"].format(method=method, weight=weight,
                                                                                        width=width, nb_plays=nb_plays)
        Evaluator.simple_rnn(units).fit(inputs, outputs).evaluate(inputs, outputs, rnn_loss_fname).predict(inputs, outputs, rnn_predictions_fname)
    if neural_type == "lstm":
        lstm_loss_fname = constants.FNAME_FORMAT["models_lstm_loss"].format(method=method, weight=weight,
                                                                            width=width, nb_plays=nb_plays)
        lstm_predictions_fname = constants.FNAME_FORMAT["models_lstm_predictions"].format(method=method, weight=weight,
                                                                                          width=width, nb_plays=nb_plays)
        Evaluator.lstm(units).fit(inputs, outputs).evaluate(inputs, outputs, lstm_loss_fname).predict(inputs, outputs, lstm_predictions_fname)

    if neural_type == "gru":
        gru_loss_fname = constants.FNAME_FORMAT["models_gru_loss"].format(method=method, weight=weight,
                                                                          width=width, nb_plays=nb_plays)
        gru_predictions_fname = constants.FNAME_FORMAT["models_gru_predictions"].format(method=method, weight=weight,
                                                                                        width=width, nb_plays=nb_plays)
        Evaluator.gru(units).fit(inputs, outputs).evaluate(inputs, outputs, gru_loss_fname).predict(inputs, outputs, gru_predictions_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-type", dest="neural_type",
                        required=False, default="simple_rnn",
                        type=str,
                        help="select which rnn neural network to evaluate")

    argv = parser.parse_args(sys.argv[1:])

    assert argv.neural_type in ["simple_rnn", "lstm", "gru"]

    args_list = [(method, weight, width, _nb_plays, argv.neural_type)
                 for method in methods
                 for weight in weights
                 for width in widths
                 for _nb_plays in nb_plays]
    # args_list = [('sin', 1, 1, 1, "simple_rnn"), ('sin', 1, 1, 2, "gru"), ('sin', 1, 1, 4, "lstm")]
    # args_list = [('sin', 1, 1, 1, "lstm")]
    # pool = pool.ProcessPool()
    # pool.starmap(loop, args_list)
    # pool.close()
    # pool.join()

    loop('sin', 1, 1, 1, 'lstm');
