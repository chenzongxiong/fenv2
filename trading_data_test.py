import trading_data
import log as logging


LOG = logging.getLogger(__name__)


if __name__ == "__main__":
    points = 5
    mu = 1
    sigma = 5

    # inputs = trading_data.DatasetGenerator.systhesis_sin_input_generator(points, mu, sigma)
    # LOG.debug("sin inputs: {}".format(inputs))
    # inputs = trading_data.DatasetGenerator.systhesis_cos_input_generator(points, mu, sigma)
    # LOG.debug("cos inputs: {}".format(inputs))

    # inputs, outputs = trading_data.DatasetGenerator.systhesis_operator_generator(points)

    # inputs, outputs = trading_data.DatasetGenerator.systhesis_play_generator(points)
    nb_plays = 3
    units = 2
    inputs, outputs = trading_data.DatasetGenerator.systhesis_model_generator(nb_plays, points, units)
    LOG.debug("inputs.shape: {}, outputs.shape: {}".format(inputs.shape, outputs.shape))

    # fname = "./tmp/trading_data_test.txt"
    # trading_data.DatasetSaver.save_data(inputs, outputs, fname)

    # inputs, outputs = trading_data.DatasetLoader.load_data(fname)
    # train_inputs, train_outputs = trading_data.DatasetLoader.load_train_data(fname)
    # test_inputs, test_outputs = trading_data.DatasetLoader.load_test_data(fname)

    # B = trading_data.DatasetGenerator.systhesis_markov_chain_generator(points, mu, sigma)
    # LOG.debug("B: {}".format(B))
