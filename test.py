import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Activation
from multiprocessing import Pool, Process, Manager
import numpy as np


class MyModel(object):
    def __init__(self, i):
        self.compiled = False
        self.id = i

    def compile(self):
        print("Start to compile Model")
        self.model = tf.keras.models.Sequential()
        self.model.add(Dense(32, input_dim=784))
        self.model.add(Activation('relu'))
        self.model.compile(optimizer='rmsprop',
                           loss='mse')
        self.compiled = True
        return self

    def fit(self):
        data = np.random.random((1000, 784))
        labels = np.random.randint(2, size=(1000, 32))
        return self.model.fit(data, labels, epochs=10, batch_size=32)


def compile(model):
    model.compile()
    model.fit()
    return np.zeros(10)

if __name__ == "__main__":
    models = [MyModel(i) for i in range(4)]
    manager = Manager()
    return_dict = manager.dict()
    # tasks = [Process(target=compile, args=(MyModel(i), return_dict)) for i in range(4)]
    # for task in tasks:
    #     task.start()

    # for task in tasks:
    #     task.join()

    # # import ipdb; ipdb.set_trace()
    # print(return_dict)

    pool = Pool(4)
    results = pool.map(compile, models)
    pool.close()
    pool.join()
    print(results)


# import trading_data as tdata
# import numpy as np
# from scipy.stats import ttest_rel
# import log as logging
# LOG = logging.getLogger(__name__)


# if __name__ == "__main__":
#     predicted_fname = './new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-50/units-20/nb_plays-20/points-1000/input_dim-1/predictions-mu-0-sigma-50-points-1000/activation#-tanh/state#-0/units#-100/nb_plays#-100/loss-mle/trends.csv'
#     a, b = tdata.DatasetLoader.load_data(predicted_fname)

#     size = a.shape[-1] - 1
#     rmse_arr1 = (a[1:]-b[1:])**2
#     rmse_arr2 = (a[1:]-a[:-1])**2

#     rmse1 = (rmse_arr1.sum()/size) ** 0.5
#     rmse2 = (rmse_arr2.sum()/size) ** 0.5
#     abs_arr1 = np.abs(a[1:]-b[1:])
#     abs_arr2 = np.abs(a[1:]-b[:-1])
#     abs1 = abs_arr1.sum()/size
#     abs2 = abs_arr2.sum()/size

#     LOG.debug("rmse1: {}".format(rmse1))
#     LOG.debug("rmse2: {}".format(rmse2))
#     LOG.debug("abs1: {}".format(abs1))
#     LOG.debug("abs2: {}".format(abs2))
#     tt1 = ttest_rel(rmse_arr1, rmse_arr2)
#     tt2 = ttest_rel(abs_arr1, abs_arr2)
#     LOG.debug("tt of rmse: {}".format(tt1))
#     LOG.debug("tt of abs: {}".format(tt2))
