import os
import json
import numpy as np
import core
import utils
import colors
import log as logging
import constants

LOG = logging.getLogger(__name__)


class DatasetGenerator(object):
    @classmethod
    def dima_sequence(cls, points=1000):
        # NOTES: only for debugging
        a = [0, 1, 5, 0, 3, 1, 5]               #dima dataset
        inputs = a*(1006 // len(a))
        inputs.insert(0, 0)
        inputs.insert(1, -100)
        inputs = np.array(inputs[:1000], dtype=np.float32)

        # inputs[600:700] = inputs[600:700] / 10.0
        # inputs[700:750] = inputs[700:750] / 7.0
        # inputs[750:800] = inputs[750:800] / 6.0
        # inputs[800:850] = inputs[800:850] / 5.0
        # inputs[850:900] = inputs[850:900] / 4.0
        # inputs[900:950] = inputs[900:950] / 3.0
        # inputs[950:975] = inputs[950:975] / 1.0
        # inputs[975:1000] = inputs[975:1000] / 0.5


        inputs[600:650] = inputs[600:650] / 10.0
        inputs[650:700] = inputs[650:700] / 7.0
        inputs[700:750] = inputs[700:750] / 6.0
        inputs[750:800] = inputs[750:800] / 5.0
        inputs[800:850] = inputs[800:850] / 4.0
        inputs[850:900] = inputs[850:900] / 3.0
        inputs[900:950] = inputs[900:950] / 1.0
        inputs[950:1000] = inputs[950:1000] / 0.5
        return inputs

    @classmethod
    def pavel_sequence(cls, points, mu=0):
        np.random.seed(0)
        inputs1 = 5.0 * np.cos(0.1 * np.linspace(-40*np.pi, 40*np.pi, points))
        inputs = (inputs1).astype(np.float32)

        sigma_list = [0.1, 0.5, 1, 2, 3, 4, 5]

        points_for_training = 600
        noise = np.zeros((points,))
        for i in range(points_for_training):
            sigma = i % len(sigma_list)
            n = np.random.normal(loc=mu, scale=sigma, size=1).astype(np.float32)
            noise[i] = n

        chunk_size = (points - points_for_training) // len(sigma_list)
        remaining = (points - points_for_training) % (len(sigma_list))
        chunk_list = [chunk_size] * len(sigma_list)

        for idx, (sigma, chunk) in enumerate(zip(sigma_list, chunk_list)):
            n = np.random.normal(loc=mu, scale=sigma, size=chunk).astype(np.float32)
            noise[600+idx*chunk:600+(idx+1)*chunk] = n

        noise[points-remaining:] = np.random.normal(loc=mu, scale=sigma_list[-1], size=remaining).astype(np.float32)
        inputs += noise

        return inputs

    @classmethod
    def systhesis_input_generator(cls, points):
        # NOTE: x = sin(t) + 3 sin(1.3 t)  + 1.2 sin (1.6 t)
        inputs1 = np.sin(np.linspace(-2*np.pi, 2*np.pi, points))
        # inputs2 = 3 * np.sin(1.3* np.linspace(-2*np.pi, 2*np.pi, points))
        # inputs3 = 1.2 * np.sin(1.6 * np.linspace(-2*np.pi, 2*np.pi, points))
        # inputs = (inputs1 + inputs2 + inputs3).astype(np.float32)
        inputs = (inputs1).astype(np.float32)
        LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = sin(t) + 3 sin(1.3 t)  + 1.2 sin (1.6 t)]")))
        return inputs

    @classmethod
    def systhesis_sin_input_generator(cls, points, mu=0, sigma=0.01, with_noise=False):
        # NOTE: x = sin(t) + 0.3 sin(1.3 t)  + 1.2 sin (1.6 t)
        inputs1 = 1.0 * np.cos(0.1 * np.linspace(-40*np.pi, 40*np.pi, points))
        # inputs1 = np.sin(np.linspace(-2*np.pi, 2*np.pi, points))
        inputs2 = 0.3 * np.sin(1.3 * np.linspace(-2*np.pi, 2*np.pi, points))
        inputs3 = 1.2 * np.sin(0.6 * np.linspace(-2*np.pi, 2*np.pi, points))

        # inputs1 = np.sin(2 * np.linspace(-10*np.pi, 10*np.pi, points))
        # inputs2 = 0.7 * np.sin(1.3 * np.linspace(-10*np.pi, 10*np.pi, points))
        # inputs3 = 1.5 * np.sin(0.6 * np.linspace(-10*np.pi, 10*np.pi, points))

        inputs = (inputs1 + inputs2 + inputs3).astype(np.float32)
        LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = sin(t) + 0.3 sin(1.3 t)  + 1.2 sin (1.6 t)]")))
        if with_noise is True:
            # import ipdb; ipdb.set_trace()
            # sigma = 5 * sigma
            # sigma = np.abs(sigma * np.cos(0.1 * np.linspace(-10 * np.pi, 10 * np.pi, points))) + 1e-3
            inputs = (inputs1).astype(np.float32)
            # NOTES: sigma = 8 + inputs2

            # noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
            # inputs += noise

            # debuging
            inputs *= 5.0
            sigma_list = [0.1, 0.5, 1, 2, 3, 4, 5]

            points_for_training = 600
            noise = np.zeros((points,))
            for i in range(points_for_training):
                sigma = i % len(sigma_list)
                n = np.random.normal(loc=mu, scale=sigma, size=1).astype(np.float32)
                noise[i] = n

            chunk_size = (points - points_for_training) // len(sigma_list)
            remaining = (points - points_for_training) % (len(sigma_list))
            chunk_list = [chunk_size] * len(sigma_list)

            for idx, (sigma, chunk) in enumerate(zip(sigma_list, chunk_list)):
                n = np.random.normal(loc=mu, scale=sigma, size=chunk).astype(np.float32)
                noise[600+idx*chunk:600+(idx+1)*chunk] = n

            noise[points-remaining:] = np.random.normal(loc=mu, scale=sigma_list[-1], size=remaining).astype(np.float32)
            inputs += noise


        return inputs

    @classmethod
    def systhesis_mixed_input_generator(cls, points, mu=0, sigma=0.01, with_noise=False):
        # NOTE: x = cos(t) + 0.7 cos(3.0 t) + 1.5 sin(2.3 t)
        # NOTE: x = cos(0.1 t) + 0.7 cos(0.2 t) + 1.5 sin(2.3 t)
        inputs1 = np.cos(0.2 * np.linspace(-10*np.pi, 10*np.pi, points))
        inputs2 = 0.7 * np.cos(2 * np.linspace(-10*np.pi, 10*np.pi, points))
        inputs3 = 1.5 * np.sin(2.3 * np.linspace(-10*np.pi, 10*np.pi, points))
        inputs = (inputs1 + inputs2 + inputs3).astype(np.float32)
        LOG.debug("Generate the input sequence according to formula {}".format(colors.red("[x = cos(t) + 0.7 cos(3.0 t)  + 1.5 sin (2.3 t)]")))
        if with_noise is True:
            noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
            inputs += noise
        return inputs

    @classmethod
    def systhesis_noise_input_generator(cls, points, mu, sigma):
        # x = np.abs(sigma * np.cos(0.1 * np.linspace(-10 * np.pi, 10 * np.pi, points))) + 1e-3
        # noise = np.random.normal(loc=mu, scale=x, size=points).astype(np.float32)
        noise = np.random.normal(loc=mu, scale=sigma, size=points).astype(np.float32)
        inputs1 = 3 * np.cos(0.1 * np.linspace(-40*np.pi, 40*np.pi, points))
        noise += (inputs1 + np.random.normal(loc=mu, scale=sigma, size=points)).astype(np.float32)
        return noise

    @classmethod
    def systhesis_operator_generator(cls,
                                     points=1000,
                                     nb_plays=1,
                                     method="sin",
                                     mu=0,
                                     sigma=0.01,
                                     with_noise=False,
                                     individual=False):
        if with_noise is True:
            if method == "sin":
                LOG.debug("Generate data with noise via sin method")
                _inputs = cls.systhesis_sin_input_generator(points, mu, sigma, with_noise=with_noise)
            elif method == "mixed":
                LOG.debug("Generate data with noise via mixed method")
                _inputs = cls.systhesis_mixed_input_generator(points, mu, sigma, with_noise_with_noise)
            elif method == "cos":
                LOG.debug("Generate data with noise via cos method")
                # _inputs = cls.systhesis_mixed_input_generator(points, mu, sigma)
                raise
        else:
            _inputs = cls.systhesis_sin_input_generator(points)

        # timestep = points
        # input_dim = 1
        timestep = 1
        input_dim = points
        operator = core.MyModel(nb_plays=nb_plays,
                                debug=True,
                                activation=None,
                                optimizer=None,
                                timestep=timestep,
                                input_dim=input_dim,
                                diff_weights=True,
                                network_type=constants.NetworkType.OPERATOR
                                )
        if individual is True:
            _outputs, multi_outputs = operator.predict(_inputs, individual=True)
            return _inputs, _outputs, multi_outputs
        else:
            _outputs = operator.predict(_inputs, individual=False)
            return _inputs, _outputs, None

    @classmethod
    def systhesis_play_generator(cls, points=1000, inputs=None):
        if inputs is None:
            _inputs = cls.systhesis_input_generator(points)
        else:
            _inputs = inputs

        play = core.Play(debug=True,
                         network_type=constants.NetworkType.PLAY)

        _outputs = play.predict(_inputs)
        _outputs = _outputs.reshape(-1)
        return _inputs, _outputs

    @classmethod
    def systhesis_model_generator(cls,
                                  inputs=None,
                                  nb_plays=1,
                                  points=1000,
                                  units=1,
                                  mu=0,
                                  sigma=0.01,
                                  input_dim=1,
                                  activation=None,
                                  with_noise=True,
                                  method=None,
                                  diff_weights=False,
                                  individual=False):

        if inputs is not None:
            points = inputs.shape[-1]

        if points % input_dim != 0:
            raise Exception("ERROR: timestep must be integer")

        # timestep = points // input_dim
        input_dim = points
        timestep = 1

        if inputs is None:
            LOG.debug("systhesis model outputs by *online-generated* inputs with settings: method: {} and noise: {}".format(colors.red(method), with_noise))

            if method == 'noise':
                _inputs = cls.systhesis_noise_input_generator(points, mu, sigma)
            elif method == 'sin':
                _inputs = cls.systhesis_sin_input_generator(points, mu, sigma, with_noise=with_noise)
            elif method == 'mixed':
                _inputs = cls.systhesis_mixed_input_generator(points, mu, sigma, with_noise=with_noise)
            elif method == 'debug-pavel':
                _inputs = cls.pavel_sequence(points, mu)
            elif method == 'debug-dima':
                _inputs = cls.dima_sequence(points)
            else:
                raise
        else:
            LOG.debug("systhesis model outputs by *pre-defined* inputs")
            _inputs = inputs

        model = core.MyModel(nb_plays=nb_plays,
                             units=units,
                             debug=True,
                             activation=activation,
                             timestep=timestep,
                             input_dim=input_dim,
                             diff_weights=diff_weights,
                             network_type=constants.NetworkType.PLAY,
                             parallel_prediction=True)

        model._make_batch_input_shape(_inputs)

        outputs, individual_outputs = model.predict_parallel(_inputs, individual=True)
        _outputs = outputs.reshape(-1)
        if individual is True:
            return _inputs, _outputs, individual_outputs
        return _inputs, _outputs

    @staticmethod
    def systhesis_markov_chain_generator(points, mu, sigma, b0=0):
        B = [b0]
        for i in range(points-1):
            bi = np.random.normal(loc=B[-1] + mu, scale=sigma)
            B.append(bi)

        return np.array(B).reshape(-1).astype(np.float32)


class DatasetLoader(object):
    SPLIT_RATIO = 0.6
    _CACHED_DATASET = {}

    @classmethod
    def load_data(cls, fname):
        if fname in cls._CACHED_DATASET:
            return cls._CACHED_DATASET[fname]

        data = np.loadtxt(fname, skiprows=0, delimiter=",", dtype=np.float32)
        inputs, outputs = data[:, 0], data[:, 1:].T
        assert len(inputs.shape) == 1
        if len(outputs.shape) == 2:
            n, d = outputs.shape
            if n == 1:
                outputs = outputs.reshape(d,)
            elif d == 1:
                outputs = outputs.reshape(n,)
            elif d == inputs.shape[0]:
                outputs = outputs.T

        cls._CACHED_DATASET[fname] = (inputs, outputs)
        return inputs, outputs

    @classmethod
    def load_train_data(cls, fname):
        if fname in cls._CACHED_DATASET:
            inputs, outputs = cls._CACHED_DATASET[fname]
            LOG.debug("Load train dataset {} from cache".format(colors.red(fname)))
        else:
            inputs, outputs = cls.load_data(fname)
            cls._CACHED_DATASET[fname] = (inputs, outputs)

        split_index = int(cls.SPLIT_RATIO * inputs.shape[0])
        train_inputs, train_outputs = inputs[:split_index], outputs[:split_index]
        return train_inputs, train_outputs

    @classmethod
    def load_test_data(cls, fname):
        if fname in cls._CACHED_DATASET:
            inputs, outputs = cls._CACHED_DATASET[fname]
            LOG.debug("Load test dataset {} from cache".format(colors.red(fname)))
        else:
            inputs, outputs = cls.load_data(fname)
            cls._CACHED_DATASET[fname] = (inputs, outputs)

        split_index = int(cls.SPLIT_RATIO * inputs.shape[0])
        test_inputs, test_outputs = inputs[split_index:], outputs[split_index:]
        return test_inputs, test_outputs


class DatasetSaver(object):
    @staticmethod
    def save_data(inputs, outputs, fname):
        assert len(inputs.shape) == 1, "length of inputs.shape must be equal to 1."
        assert inputs.shape[0] == outputs.shape[0], \
          "inputs.shape[0] is: {}, whereas outputs.shape[0] is {}.".format(inputs.shape[0], outputs.shape[0])
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)
        if len(outputs.shape) == 1:
            outputs = outputs.reshape(-1, 1)

        res = np.hstack([inputs, outputs])
        np.savetxt(fname, res, fmt="%.3f", delimiter=",")


    @staticmethod
    def save_loss(loss, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w") as f:
            f.write(json.dumps(loss))
