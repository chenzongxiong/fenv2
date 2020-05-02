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
                        action='store_true')
    parser.add_argument('--sigma', dest='sigma',
                        required=True)

    parser.add_argument('--activation', dest='activation',
                        default=None,
                        required=False)

    parser.add_argument('--method', dest='method',
                        # default=None,
                        required=True)


    argv = parser.parse_args(sys.argv[1:])

    # method = 'sin'
    # method = 'debug-pavel'
    method = argv.method
    state = 0
    input_dim = 1

    if argv.markov_chain:
        activation = None
        nb_plays = 1
        mu = 0
        sigma = 110
        points = 1000
        batch_size = 2000
        __activation__LIST = ['elu']

        # __nb_plays__LIST_units10000_nb_plays20 = [25, 50, 50]
        # __units__LIST_units10000_nb_plays20 = [25, 50, 100]

        __nb_plays__LIST_units10000_nb_plays20 = [25]
        __units__LIST_units10000_nb_plays20 = [25]

        __units__LIST = [__units__LIST_units10000_nb_plays20]
        __nb_plays__LIST = [__nb_plays__LIST_units10000_nb_plays20]

        nb_plays_LIST = [20]
    elif argv.diff_weights:
        activation = argv.activation
        nb_plays = 1
        mu = 0
        sigma = argv.sigma
        points = 1000
        __activation__LIST = ['tanh', 'elu']
        __activation__LIST = ['elu']

        __units__LIST_units1_nb_plays1 = []
        __nb_plays__LIST_units1_nb_plays1 = []

        __units__LIST_units50_nb_plays50 = [10, 10, 25, 50, 50]
        __nb_plays__LIST_units50_nb_plays50 = [10, 25, 25, 50, 100]

        # __units__LIST_units100_nb_plays100 = [25, 25, 50, 50, 100, 25, 100]
        # __nb_plays__LIST_units100_nb_plays100 = [25, 50, 50, 100, 100, 200, 200]

        __units__LIST_units100_nb_plays100 = []
        __nb_plays__LIST_units100_nb_plays100 = []


        # __units__LIST_units100_nb_plays500 = [25, 25, 50, 50, 100, 100, 100, 200]
        # __nb_plays__LIST_units100_nb_plays500 = [25, 50, 50, 100, 100, 200, 500, 500]


        # __units__LIST = [__units__LIST_units50_nb_plays50,
        #                 __units__LIST_units100_nb_plays100,
        #                 __units__LIST_units100_nb_plays500]
        # __nb_plays__LIST = [__nb_plays__LIST_units50_nb_plays50,
        #                     __nb_plays__LIST_units100_nb_plays100,
        #                     __nb_plays__LIST_units100_nb_plays500]
        # nb_plays_LIST = [50, 100, 500]

        __units__LIST = [__units__LIST_units50_nb_plays50,
                         __units__LIST_units100_nb_plays100]
        __nb_plays__LIST = [__nb_plays__LIST_units50_nb_plays50,
                            __nb_plays__LIST_units100_nb_plays100]
        # nb_plays_LIST = [50, 100]
        nb_plays_LIST = [50]
    else:
        pass

        nb_plays_LIST = [1, 50, 100, 500]

    lr = 0.05
    epochs = 10000
    NUM_ROWS_TO_READ = 500
    ensemble_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    overview = []
    split_ratio = 0.6


    if True:
        __nb_plays__ = 25
        __units__ = 25
        __state__ = 0
        excel_fname = './new-dataset/models/diff_weights/method-{}/hnn-mse-sigma-{}-nb_plays#-{}-units#-{}-__state__-{}.xlsx'.format(method, sigma, __nb_plays__, __units__, __state__)

    writer = pd.ExcelWriter(excel_fname, engine='xlsxwriter')

    for idx, nb_plays in enumerate(nb_plays_LIST):
        diff_frame = pd.DataFrame({})
        normalizedframe = pd.DataFrame({})
        lossframe = pd.DataFrame({})
        units = nb_plays
        if nb_plays == 500:
            units = 100
        if True:
            input_fname = constants.DATASET_PATH['models_diff_weights'].format(method=method, activation=activation, state=state, mu=mu, sigma=sigma, units=units, nb_plays=nb_plays, points=points, input_dim=input_dim)
            base = pd.read_csv(input_fname, header=None, names=['inputs', 'outputs'], skiprows=int(0.6*points))

        dataframe = base.copy(deep=False)

        _base_output = base['outputs'].values
        _base_diff = _base_output[1:] - _base_output[:-1]
        _base_mu = _base_diff.mean()
        _base_sigma = _base_diff.std()

        normalizedframe = base.copy(deep=False)
        diff_frame = diff_frame.assign(outputs=(_base_output[1:] - _base_output[:-1] - _base_mu) / _base_sigma)

        for __activation__ in __activation__LIST:

            for ensemble in ensemble_LIST:
                __nb_plays__ = 25
                __units__ = 25

                LOG.debug(colors.cyan("Loading from diff weights..."))
                prediction_fname = constants.DATASET_PATH['models_diff_weights_predictions'].format(method=method,
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
                                                                                                    ensemble=ensemble,
                                                                                                    __units__=__units__,
                                                                                                    __nb_plays__=__nb_plays__,
                                                                                                    loss='mse')

                LOG.debug(colors.cyan("input file: {}".format(input_fname)))
                LOG.debug(colors.cyan("predict file: {}".format(prediction_fname)))

                predict_column = 'nb_plays-{}-units-{}-__activation__-{}-__nb_plays__-{}-__units__-{}-ensemble-{}=predictions'.format(nb_plays, units, __activation__, __nb_plays__, __units__, ensemble)
                prediction = pd.read_csv(prediction_fname, header=None, names=['inputs', predict_column], nrows=NUM_ROWS_TO_READ)

                kwargs = {predict_column: prediction[predict_column]}
                dataframe = dataframe.assign(**kwargs)

                loss = ((prediction[predict_column]  - base['outputs']).values[50:] ** 2).mean()**0.5
                number_of_parameters = __nb_plays__ * (2*__units__ + __units__) + 1

                _predict_output = prediction[predict_column].values
                _predict_diff = _predict_output[1:] - _predict_output[:-1]
                _predict_mu = _predict_diff.mean()
                _predict_sigma = _predict_diff.std()
                diff_rmse = ((_predict_diff - _base_diff) ** 2).mean() ** 0.5
                if argv.markov_chain:
                    overview.append([activation, __activation__, units, __nb_plays__, __units__, lr, epochs, loss, number_of_parameters, _base_mu, _predict_mu, _base_sigma, _predict_sigma, diff_rmse, ensemble, loss_list['loss'].values[-1]])
                else:
                    overview.append([activation, __activation__, units, __nb_plays__, __units__, lr, epochs, loss, number_of_parameters, _base_mu, _predict_mu, _base_sigma, _predict_sigma, diff_rmse])

                if _predict_mu * _base_mu >= 0:
                    normalized_data = (prediction[predict_column].values - _predict_mu) / _predict_sigma
                else:
                    normalized_data = - (prediction[predict_column].values - _predict_mu) / _predict_sigma
                kwargs = {predict_column: normalized_data}
                normalizedframe = normalizedframe.assign(**kwargs)

                kwargs = {'nb_plays-{}-units-{}-__activation__-{}-__nb_plays__-{}-__units__-{}-ensemble-{}-diff'.format(nb_plays, units, __activation__, __nb_plays__, __units__, ensemble): (_predict_output[1:] - _predict_output[:-1] - _predict_mu) / _predict_sigma}
                diff_frame = diff_frame.assign(**kwargs)


        dataframe.to_excel(writer, sheet_name="nb_plays-{}-pred".format(nb_plays, index=False))
        normalizedframe.to_excel(writer, sheet_name="nb_plays-{}-n-pred".format(nb_plays, index=False))
        diff_frame.to_excel(writer, sheet_name="nb_plays-{}-diff".format(nb_plays, index=False))

    # if argv.markov_chain:
    #     overview = pd.DataFrame(overview,
    #                             columns=['activation', '__activation__', 'nb_plays/units', '__nb_plays__', '__units__', 'adam_learning_rate', 'epochs', 'rmse', 'nb_paramters', 'base_mu', 'predict_mu', 'base_sigma', 'predict_sigma', 'diff_rmse', 'ensemble', 'min-loss'])

    # else:
    if True:
        overview = pd.DataFrame(overview,
                                columns=['activation', '__activation__', 'nb_plays/units', '__nb_plays__', '__units__', 'adam_learning_rate', 'epochs', 'rmse', 'nb_paramters', 'base_mu', 'predict_mu', 'base_sigma', 'predict_sigma', 'diff_rmse'])

    overview.to_excel(writer, sheet_name='overview', index=False)
    writer.close()
