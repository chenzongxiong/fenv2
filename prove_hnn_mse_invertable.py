import sys
import time
import argparse
import numpy as np

from core import MyModel
import utils
import trading_data as tdata
import log as logging
import constants
import colors

sess = utils.get_session()
LOG = logging.getLogger(__name__)
epochs = constants.EPOCHS
EPOCHS = constants.EPOCHS


def fit(inputs,
        outputs,
        units=1,
        activation='tanh',
        nb_plays=1,
        learning_rate=0.001,
        loss_file_name="./tmp/my_model_loss_history.csv",
        weights_name='model.h5',
        epochs=1000):

    # steps_per_epoch = batch_size

    start = time.time()
    # input_dim = 1
    # timestep = inputs.shape[0] // input_dim
    timestep = 1
    input_dim = inputs.shape[0]

    steps_per_epoch = 1

    mymodel = MyModel(input_dim=input_dim,
                      timestep=timestep,
                      units=units,
                      activation=activation,
                      nb_plays=nb_plays)
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

    mymodel.save_weights(weights_fname)


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
                      parallel_prediction=True)

    mymodel.load_weights(weights_fname)
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
    parser.add_argument("--__nb_plays__", dest="__nb_plays__",
                        required=False,
                        type=int)
    parser.add_argument("--__units__", dest="__units__",
                        required=False,
                        type=int)
    parser.add_argument("--__activation__", dest="__activation__",
                        required=False,
                        type=str)
    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")
    parser.add_argument('--force_train', dest='force_train',
                        required=False,
                        action="store_true")

    argv = parser.parse_args(sys.argv[1:])


    loss_name = 'mse'
    method = 'sin'
    input_dim = 1
    state = 0
    __state__ = 0
    ############################## Misc #############################
    mu = int(argv.mu)
    sigma = int(argv.sigma)

    points = argv.points
    epochs = argv.epochs
    force_train = argv.force_train
    learning_rate = argv.lr
    ############################## ground truth #############################
    nb_plays = argv.nb_plays
    units = argv.units
    activation = argv.activation
    ############################## predicitons #############################
    __nb_plays__ = argv.__nb_plays__
    __units__ = argv.__units__
    __activation__ = argv.__activation__

    # # Hyper Parameters

    input_fname_key = 'models_diff_weights' if argv.diff_weights else 'models'
    predict_fname_key = 'models_diff_weights_invert_predictions' if argv.diff_weights else 'models_invert_predictions'
    loss_history_fname_key = 'models_diff_weights_invert_loss_history' if argv.diff_weights else 'models_invert_loss_history'
    weight_fname_key = 'models_diff_weights_invert_saved_weights' if argv.diff_weights else 'models_invert_saved_weights'
    input_fname = constants.DATASET_PATH[input_fname_key].format(method=method,
                                                                 activation=activation,
                                                                 state=state,
                                                                 mu=mu,
                                                                 sigma=sigma,
                                                                 units=units,
                                                                 nb_plays=nb_plays,
                                                                 points=points,
                                                                 input_dim=input_dim)

    loss_history_fname = constants.DATASET_PATH[loss_history_fname_key].format(method=method,
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

    predict_fname = constants.DATASET_PATH[predict_fname_key].format(method=method,
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

    weights_fname = constants.DATASET_PATH[weight_fname_key].format(method=method,
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

    LOG.debug("==================== INFO ====================")
    LOG.debug(colors.red("Test multiple plays"))
    LOG.debug(colors.cyan("input_fname: {}".format(input_fname)))
    LOG.debug(colors.cyan("predict_fname: {}".format(predict_fname)))
    LOG.debug(colors.cyan("loss_history_file: {}".format(loss_history_fname)))
    LOG.debug(colors.cyan("weights_fname: {}".format(weights_fname)))
    LOG.debug(colors.cyan("learning rate: {}".format(learning_rate)))
    LOG.debug(colors.cyan("points: {}".format(points)))
    LOG.debug(colors.cyan("method: {}".format(method)))
    LOG.debug(colors.cyan("force_train: {}".format(force_train)))
    LOG.debug("==============================================")

    train_inputs, train_outputs = tdata.DatasetLoader.load_train_data(input_fname)
    test_inputs, test_outputs = tdata.DatasetLoader.load_test_data(input_fname)

    import ipdb; ipdb.set_trace()

    fit(inputs=train_outputs,
        outputs=train_inputs,
        units=__units__,
        activation=__activation__,
        nb_plays=__nb_plays__,
        learning_rate=learning_rate,
        loss_file_name=loss_history_fname,
        weights_name=weights_fname,
        epochs=epochs)

    test_inputs, predictions = predict(inputs=[train_outputs, test_outputs],
                                       outputs=[train_inputs, test_inputs],
                                       units=__units__,
                                       activation=__activation__,
                                       nb_plays=__nb_plays__,
                                       weights_name=weights_fname)

    tdata.DatasetSaver.save_data(test_inputs, predictions, predict_fname)
