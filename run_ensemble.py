import core
import trading_data as tdata


if __name__ == '__main__':
    # best_epoch_list = [6000, 6000, 1000, 1000, 1000]
    ensembles = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    ensembles = [11, 14, 15, 16, 19]

    best_epoch_list = [19000] * len(ensembles)
    batch_size = input_dim = 1500
    timestep = 1
    nb_plays = 100
    units = 100
    activation = 'elu'
    parallel_prediction = True
    init_at_once = False
    ensemble_models = core.EnsembleModel(ensembles,
                                         input_dim,
                                         timestep,
                                         units,
                                         activation,
                                         nb_plays,
                                         parallel_prediction,
                                         best_epoch_list=best_epoch_list)
    ensemble_models.load_weights()
    mu = 0
    sigma = 110
    fname = './new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv'
    # price, noise = tdata.DatasetLoader.load_data(fname)
    # ensemble_models.plot_graphs_together(price[:1500], noise[:1500], mu, sigma)
    ensemble_models.trend()
