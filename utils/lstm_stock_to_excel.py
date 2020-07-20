import sys
import json
sys.path.append('.')
sys.path.append('..')

import argparse

import pandas as pd
import constants
import log as logging

import colors


LOG = logging.getLogger(__name__)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--diff-weights', dest='diff_weights',
                        required=False,
                        action="store_true")
    parser.add_argument('--markov-chain', dest='markov_chain',
                        required=False,
                        action="store_true")

    parser.add_argument('--method', dest='method',
                        required=False,
                        type=str,
                        default='sin')

    # diff weights: sigma = 8,
    # sigma = 2

    parser.add_argument('--sigma', dest='sigma',
                        required=True)

    argv = parser.parse_args(sys.argv[1:])


    method = argv.method
    state = 0
    input_dim = 1

    nb_plays = 1
    units = 1
    mu = 0
    sigma = int(argv.sigma)
    points = 0

    # LOG.debug("====================INFO====================")
    # LOG.debug(colors.cyan("units: {}".format(units)))
    # LOG.debug(colors.cyan("__units__: {}".format(__units__)))
    # # LOG.debug(colors.cyan("method: {}".format(method)))
    # LOG.debug(colors.cyan("nb_plays: {}".format(nb_plays)))
    # # LOG.debug(colors.cyan("input_dim: {}".format(input_dim)))
    # # LOG.debug(colors.cyan("state: {}".format(state)))
    # LOG.debug(colors.cyan("mu: {}".format(mu)))
    # LOG.debug(colors.cyan("sigma: {}".format(sigma)))
    # LOG.debug(colors.cyan("activation: {}".format(activation)))
    # LOG.debug(colors.cyan("points: {}".format(points)))
    # LOG.debug(colors.cyan("epochs: {}".format(epochs)))
    # LOG.debug(colors.cyan("Write  data to file {}".format(input_fname)))
    # LOG.debug("================================================================================")
    input(colors.red("RUN script ./lstm_loss_history_collector.sh #diff_weights before run this script"))
    __units__LIST = [1, 8, 16, 32, 64, 128, 256]
    if argv.diff_weights:
        # nb_plays_LIST = [50, 100, 500]
        nb_plays_LIST = [50]
        activation = 'tanh'
        if method != 'stock':
            raise Exception("method must be stock")

        if method == 'stock':
            activation = None
            nb_plays_LIST = [0]

    if argv.markov_chain:
        lr = 0.005
    else:
        lr = 0.001

    epochs = 30000

    overview = []
    split_ratio = 0.6

    __activation__ = 'tanh'

    if argv.diff_weights:
        __units__ = 64
        excel_fname = './new-dataset/lstm/diff_weights/method-{}/lstm-all-sigma-{}-units-{}.xlsx'.format(method, sigma, __units__)

    writer = pd.ExcelWriter(excel_fname, engine='xlsxwriter')

    # ensemble_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ensemble_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for nb_plays in nb_plays_LIST:
        lossframe = pd.DataFrame({})
        normalizedframe = pd.DataFrame({})
        diff_frame = pd.DataFrame({})
        units = nb_plays

        if argv.diff_weights:
            input_fname = constants.DATASET_PATH['models_diff_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)

        base = pd.read_csv(input_fname, header=None, names=['inputs', 'outputs'], skiprows=1300, nrows=400)
        # import ipdb; ipdb.set_trace()
        dataframe = base.copy(deep=False)

        _base_output = base['outputs'].values
        _base_diff = _base_output[1:] - _base_output[:-1]
        _base_mu = _base_diff.mean()
        _base_sigma = _base_diff.std()

        normalizedframe = base.copy(deep=False)
        # normalizedframe = normalizedframe.assign(outputs=(_base_output-_base_output.mean())/_base_output.std())
        # diff_frame = diff_frame.assign(outputs=(_base_output[1:] - _base_output[:-1] - _base_mu) / _base_sigma)

        diff_frame = diff_frame.assign(outputs=(_base_output[1:] - _base_output[:-1]))
        __units__ = 64
        for ensemble in ensemble_LIST:
        # for __units__ in __units__LIST:
            # ensemble = 1

            if argv.diff_weights:
                prediction_fname = constants.DATASET_PATH['lstm_diff_weights_prediction'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss='mle', ensemble=ensemble)
                loss_fname = constants.DATASET_PATH['lstm_diff_weights_loss'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, loss='mle', ensemble=ensemble)
                loss_file_fname = constants.DATASET_PATH['lstm_diff_weights_loss_file'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim, __units__=__units__, learning_rate=lr, loss='mle', ensemble=ensemble)

            LOG.debug(colors.cyan("input file: {}".format(input_fname)))
            LOG.debug(colors.cyan("predict file: {}".format(prediction_fname)))
            LOG.debug(colors.cyan("loss file: {}".format(loss_file_fname)))

            predict_column = 'nb_plays-{}-units-{}-ensemble-{}-predictions'.format(nb_plays, __units__, ensemble)
            prediction = pd.read_csv(prediction_fname, header=None, names=['inputs', predict_column])
            loss_list = pd.read_csv(loss_file_fname, header=None, names=['loss'])

            # import ipdb; ipdb.set_trace()
            prediction, predict_mu, predict_sigma = prediction[:-2], prediction[predict_column].values[-2], prediction[predict_column].values[-1]
            kwargs = {predict_column: prediction[predict_column]}
            dataframe = dataframe.assign(**kwargs)
            kwargs = {"nb_plays-{}-units-{}-ensemble-{}-loss".format(nb_plays, __units__, ensemble): loss_list['loss']}
            lossframe = lossframe.assign(**kwargs)

            with open(loss_fname) as f:
                loss = json.loads(f.read())

            rmse1 = ((base['outputs'] - prediction[predict_column]).values ** 2).mean() ** 0.5
            rmse2 = ((base['outputs'] + prediction[predict_column]).values ** 2).mean() ** 0.5
            rmse = min(rmse1, rmse2)
            # https://stackoverflow.com/questions/38080035/how-to-calculate-the-number-of-parameters-of-an-lstm-network
            number_of_parameters = 4 * ((1 + 1) * __units__ + __units__*__units__)

            _predict_output = prediction[predict_column].values
            _predict_diff = _predict_output[1:] - _predict_output[:-1]
            _predict_mu = _predict_diff.mean()
            _predict_sigma = _predict_diff.std()
            diff_rmse1 = ((_predict_diff - _base_diff) ** 2).mean() ** 0.5
            diff_rmse2 = ((_predict_diff + _base_diff) ** 2).mean() ** 0.5
            diff_rmse = min(diff_rmse1, diff_rmse2)
            # import ipdb; ipdb.set_trace()
            # kl = np.log()

            # import ipdb; ipdb.set_trace()
            overview.append([nb_plays, __units__, lr, epochs, rmse, loss['diff_tick'], number_of_parameters, _base_mu, _predict_mu, _base_sigma, _predict_sigma, diff_rmse, loss_list['loss'].values[-1]])

            normalized_column = 'nb_plays-{}-units-{}-ensemble-{}-predictions'.format(nb_plays, __units__, ensemble)
            # if _predict_mu * _base_mu >= 0:
            #     normalized_data = prediction[predict_column].values
            # else:
            #     normalized_data = - prediction[predict_column].values
            # normalized_data = (prediction[predict_column] - predict_mu) / predict_sigma
            # if _predict_mu * _base_mu >= 0:
            #     normalized_data = -normalized_data
            # else:
            normalized_data = prediction[predict_column].values

            if diff_rmse1 < diff_rmse2:
                normalized_data = normalized_data
            else:
                normalized_data = - normalized_data

            kwargs = {normalized_column:  normalized_data}
            normalizedframe = normalizedframe.assign(**kwargs)
            # kwargs = {'nb_plays-{}-units-{}-ensemble-{}-diff'.format(nb_plays, __units__, ensemble): (_predict_output[1:] - _predict_output[:-1] - _predict_mu) / predict_sigma}
            kwargs = {'nb_plays-{}-units-{}-ensemble-{}-diff'.format(nb_plays, __units__, ensemble): (_predict_output[1:] - _predict_output[:-1])}

            diff_frame = diff_frame.assign(**kwargs)

        dataframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-pred".format(nb_plays, '1-256', index=False))
        normalizedframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-n-pred".format(nb_plays, '1-256', index=False))
        diff_frame.to_excel(writer, sheet_name="nb_plays-{}-units-{}-diff".format(nb_plays, '1-256', index=False))
        lossframe.to_excel(writer, sheet_name="nb_plays-{}-units-{}-loss".format(nb_plays, '1-256', index=False))

    overview = pd.DataFrame(overview,
                            columns=['nb_plays/units', 'lstm_units', 'adam_learning_rate', 'epochs', 'rmse', 'time_cost_(s)', 'nb_paramters', 'ground_truth_mu', 'predict_mu', 'ground_truth_sigma', 'predict_sigma', 'diff_rmse', 'min-loss'])

    overview.to_excel(writer, sheet_name='overview', index=False)
    writer.close()
    LOG.debug("Write to excel file {}".format(excel_fname))
