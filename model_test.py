import pool
import os
import numpy as np

import constants
import log as logging


LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
epochs = 1
batch_size = constants.BATCH_SIZE


def fit(inputs, outputs, nb_plays):
    import tensorflow as tf
    import core

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    play_model = core.PlayModel(nb_plays)

    play_model.compile(loss="mse",
                       optimizer=optimizer,
                       metrics=["mse"])

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50)
    play_model.fit(inputs, outputs, epochs=epochs, verbose=1, batch_size=batch_size,
                   shuffle=False, callbacks=[early_stopping_callback])

    return play_model


def evaluate(play_model, inputs, outputs):
    loss, mse = play_model.evaluate(inputs, outputs, verbose=1, batch_size=batch_size)
    LOG.info("loss: {}, mse: {}".format(loss, mse))
    return loss, mse


def predict(play_model, inputs, debug_plays=False):

    predictions = play_model.predict(inputs, batch_size=batch_size, verbose=1)
    if debug_plays:
        plays_outputs = play_model.get_plays_outputs(inputs)
        sess.run(play_model, feed_dict={"inputs": inputs})
        return predictions, plays_outputs

    return predictions, None


def loop(method, weight, width, bz, nb_plays):
    import trading_data as tdata

    LOG.debug("pid: {}, method: {}, weight: {}, width: {}, batch_size: {}, nb_plays: {}".format(
      os.getpid(), method, weight, width, bz, nb_plays))

    fname = constants.FNAME_FORMAT["models"].format(method=method, weight=weight,
                                                    width=width, nb_plays=nb_plays)

    train_inputs, train_outputs = tdata.DatasetLoader.load_train_data(fname)
    test_inputs, test_outputs = tdata.DatasetLoader.load_test_data(fname)

    _train_inputs = train_inputs.reshape(-1, bz)  # samples * sequences
    _train_outputs = train_outputs.reshape(-1, bz)  # samples * sequences
    _test_inputs = test_inputs.reshape(-1, bz)  # samples * sequences
    _test_outputs = test_outputs.reshape(-1, bz)  # samples * sequences

    for nb_plays_ in constants.NB_PLAYS:
        LOG.debug("Fitting...")
        play_model = fit(_train_inputs, _train_outputs, nb_plays_)
        LOG.debug("Evaluating...")
        loss, mse = evaluate(play_model, _test_inputs, _test_outputs)
        fname = constants.FNAME_FORMAT["models_loss"].format(method=method, weight=weight,
                                                             width=width, nb_plays=nb_plays_,
                                                             batch_size=bz)
        tdata.DatasetSaver.save_loss({"loss": loss, "mse": mse}, fname)
        train_predictions, train_plays_outputs = predict(play_model, _train_inputs, debug_plays=False)
        test_predictions, test_plays_outputs = predict(play_model, _test_inputs, debug_plays=False)

        train_predictions = train_predictions.reshape(-1)
        test_predictions = test_predictions.reshape(-1)

        inputs = np.hstack([train_inputs, test_inputs])
        predictions = np.hstack([train_predictions, test_predictions])
        fname = constants.FNAME_FORMAT["models_predictions"].format(method=method, weight=weight,
                                                                    width=width, nb_plays=nb_plays,
                                                                    nb_plays_=nb_plays,
                                                                    batch_size=bz)
        tdata.DatasetSaver.save_data(inputs, predictions, fname)
        if train_plays_outputs and test_plays_outputs:
            fname = constants.FNAME_FORMAT["models_multi_predictions"].format(method, weight=weight,
                                                                              width=width, nb_plays=nb_plays,
                                                                              nb_plays_=nb_plays,
                                                                              batch_size=bz)
            plays_outputs = np.vstack([train_plays_outputs, test_plays_outputs])
            tdata.DatasetSaver.save_data(inputs, plays_outputs, fname)


if __name__ == "__main__":
    methods = constants.METHODS
    weights = constants.WEIGHTS
    widths = constants.WIDTHS
    nb_plays = constants.NB_PLAYS
    batch_size_list = constants.BATCH_SIZE_LIST

    args_list = [(method, weight, width, bz, _nb_plays)
                for method in methods
                for weight in weights
                for width in widths
                for bz in batch_size_list
                for _nb_plays in nb_plays]

    # args_list = [('sin', 1, 1, 4, 1), ('sin', 1, 1, 4, 2), ('sin', 1, 1, 4, 3), ('sin', 1, 1, 4, 4), ('sin', 1, 1, 4, 8)]
    args_list = [('sin', 1, 1, 4, 1)]
    _pool = pool.ProcessPool()
    _pool.starmap(loop, args_list)
