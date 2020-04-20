# import utils
# import constants
# import log as logging
# import core

# LOG = logging.getLogger(__name__)



# nb_plays = 20
# # weights_file_key = 'models_diff_weights_saved_weights'
# weights_file_key = 'models_diff_weights_mc_saved_weights'
# method = 'sin'
# loss_name = 'mse'
# loss_name = 'mle'

# mu = 0
# sigma = 50
# points = 1000
# input_dim = 1
# # ground truth
# nb_plays = 20
# units = 20
# state = 0
# activation = None
# # activation = 'tanh'
# # predicitons
# __nb_plays__ = 20
# __units__ = 20
# __state__ = 0
# __activation__ = None
# # __activation__ = 'tanh'
# __activation__ = 'relu'


# weights_fname = constants.DATASET_PATH[weights_file_key].format(method=method,
#                                                                 activation=activation,
#                                                                 state=state,
#                                                                 mu=mu,
#                                                                 sigma=sigma,
#                                                                 units=units,
#                                                                 nb_plays=nb_plays,
#                                                                 points=points,
#                                                                 input_dim=input_dim,
#                                                                 __activation__=__activation__,
#                                                                 __state__=__state__,
#                                                                 __units__=__units__,
#                                                                 __nb_plays__=__nb_plays__,
#                                                                 loss=loss_name)


# def show_weights(weights_fname):
#     for i in range(nb_plays):
#         LOG.debug("==================== PLAY {} ====================".format(i+1))
#         fname = weights_fname[:-3] + '/{}plays/play-{}.h5'.format(nb_plays, i)
#         LOG.debug("Fname: {}".format(fname))
#         utils.read_saved_weights(fname)

# def show_loss():
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#     import numpy as np
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')

#     phi_weight= np.linspace(0, 5, 500)
#     # theta = np.linspace(-10, 10, 2000)
#     # bias = np.linspace(-10, 10, 2000)
#     # tilde_theta = np.linspace(-10, 10, 2000)
#     # tilde_bias = np.linspace(-10, 10, 2000)

#     # mymodel = core.MyModel()

#     x = phi_weight * np.sin(20 * phi_weight)
#     y = phi_weight * np.cos(20 * phi_weight)

#     c = x + y

#     # ax.scatter(x, y, phi_weight, c=c)
#     ax.plot(x, y, phi_weight, '-b')
#     # ax.plot(x, y, c, '-b')
#     # ax.plot_surface(x, y, phi_weight,
#     #                 cmap=plt.cm.jet,
#     #                 rstride=1,
#     #                 cstride=1,
#     #                 linewidth=0)
#     plt.show()


# if __name__ == "__main__":
#     # show_loss()
#     show_weights(weights_fname)
import trading_data as tdata

# fname1 = './new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-20/nb_plays-20/points-1000/input_dim-1/predictions-mu-0-sigma-110-points-1000/activation#-tanh/state#-0/units#-100/nb_plays#-100/loss-mle/predictions-batch_size-1000-epochs-6000.csv'
fname1 = 'new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv'
length = 1500
prices, random_walk1 = tdata.DatasetLoader.load_data(fname1)
random_walk1 = random_walk1[:length]
noise1 = random_walk1[1:] - random_walk1[:-1]
# import ipdb; ipdb.set_trace()
# fname2 = './new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-20/nb_plays-20/points-1000/input_dim-1/predictions-mu-0-sigma-110-points-1000/activation#-tanh/state#-0/units#-100/nb_plays#-100/loss-mle/predictions-batch_size-1000.csv'
# fname2 = '/Users/zxchen/Desktop/predictions-elu-batch_size-300-epochs-8000.csv'
# fname2 = '/Users/zxchen/Desktop/predictions-elu-batch_size-300-epochs-20000.csv'
name = 'predictions-elu-batch_size-1500-epochs-16000-ensemble-6'
fname2 = '/Users/zxchen/Desktop/{}.csv'.format(name)
prices, random_walk2 = tdata.DatasetLoader.load_data(fname2)
random_walk2 = random_walk2[:length]
noise2 = random_walk2[1:] - random_walk2[:-1]

tdata.DatasetSaver.save_data(noise1, noise2, '/Users/zxchen/Desktop/diff-{}.csv'.format(name))
