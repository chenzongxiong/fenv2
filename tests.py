import unittest
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.ops.parallel_for.gradients import jacobian
import pickle
import pickletools

import core
import utils


class TestCases(unittest.TestCase):
    def setUp(self):
        self.session = utils.get_session()
        self.inputs = np.array([1, 1.5, 2.5, 2.5, -0.5, -0.25, -1, 0.25, 1/3.0, 0.1,
                                0, 0.21, -1.5, 0.7, 0.9, 1.5, -0.4, 1, -0.15, 2])
        self.truth = np.array([0.5, 1, 2, 2, 0, 0, -0.5, -0.25, -1/6, -1/6,
                               -1/6, -1/6, -1, 0.2, 0.4, 1, 0.1, 0.5, 0.35, 1.5])

        self.truth_with_state_zero = self.truth
        self.truth_with_state_one = np.array([1, 1, 2, 2, 0, 0, -0.5, -0.25, -1/6, -1/6,
                                              -1/6, -1/6, -1, 0.2, 0.4, 1, 0.1, 0.5, 0.35, 1.5])

    def tearDown(self):
        # utils.clear_session()
        pass

    # def test_Phi(self):
    #     a = tf.constant([[1]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))
    #     self.assertTrue(aa[0, 0] == 0.5)

    #     a = tf.constant([[0.5]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))
    #     self.assertTrue(aa[0, 0] == 0)

    #     a = tf.constant([[-0.5]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))
    #     self.assertTrue(aa[0, 0] == 0)

    #     a = tf.constant([[-1]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))
    #     self.assertTrue(aa[0, 0] == -0.5)

    #     a = tf.constant([[0]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))

    #     self.assertTrue(aa[0, 0] == 0)

    #     a = tf.constant([[100]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))
    #     self.assertTrue(aa[0, 0] == 99.5)

    #     a = tf.constant([[-100]], dtype=tf.float32, name="a")
    #     aa = self.session.run(core.Phi(a))
    #     self.assertTrue(aa.shape == (1, 1))
    #     self.assertTrue(aa[0, 0] == -99.5)

    # def test_phi(self):
    #     a = np.array([1], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == 0.5)

    #     a = np.array([-1], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == -0.5)

    #     a = np.array([0], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == 0)

    #     a = np.array([0.5], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == 0)

    #     a = np.array([-0.5], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == 0)

    #     a = np.array([100], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == 99.5)

    #     a = np.array([-100], dtype=np.float32)
    #     aa = core.phi(a)
    #     self.assertTrue(aa == -99.5)

    def test_phicell(self):
        # NOTE: failed
        pass
        # import ipdb; ipdb.set_trace()
        # input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        # cell = core.PhiCell(debug=True)
        # output_1 = cell(input_1, [[[0]]])
        # utils.init_tf_variables()
        # input_2 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
        # output_2 = cell(input_1, [[[0]]])
        # utils.init_tf_variables()

        # result_1, result_2 = self.session.run([output_1, output_2])
        # # check outputs with different input sequence
        # self.assertTrue(np.all(result_1[0].reshape(-1) == result_2[0][0].reshape(-1)))
        # # check state with different input sequence
        # self.assertTrue(result_1[1][0].reshape(-1)[0] == result_2[1][0].reshape(-1)[0])

        # # check the value of outputs
        # self.assertTrue(np.allclose(result_1[0].reshape(-1), self.truth, atol=1e-7))
        # # check the value of state
        # self.assertTrue(result_1[1][0].reshape(-1)[0] == 1.5)

    # def test_operator(self):
    #     input_1 = ops.convert_to_tensor(self.inputs.reshape([1, 1, -1]), dtype=tf.float32)
    #     operator_1 = core.Operator(debug=True)
    #     output_1 = operator_1(input_1)
    #     input_2 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
    #     operator_2 = core.Operator(debug=True)
    #     output_2 = operator_2(input_2)
    #     utils.init_tf_variables()
    #     result_1, result_2 = self.session.run([output_1, output_2])
    #     self.assertTrue(np.allclose(result_1.reshape(-1), self.truth))
    #     self.assertTrue(np.allclose(result_2.reshape(-1), self.truth))

    # def test_mydense(self):
    #     units = 10
    #     _init_bias = 1
    #     _init_kernel = 2
    #     input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
    #     mydense = core.MyDense(units=units, activation=None, use_bias=True, debug=True, _init_bias=_init_bias, _init_kernel=_init_kernel)

    #     output_1 = mydense(input_1)
    #     utils.init_tf_variables()
    #     result_1 = self.session.run(output_1)
    #     # check shape, must be equal to ()
    #     self.assertTrue(result_1.shape == (1, self.inputs.shape[0], units))
    #     # check value
    #     kernel = np.array([_init_kernel] * units).reshape(1, units)
    #     bias = np.array([_init_bias])
    #     truth = self.inputs.reshape([1, -1, 1]) * kernel + bias
    #     _truth = self.inputs * _init_kernel + _init_bias
    #     self.assertTrue(np.allclose(result_1, truth, atol=1e-7))
    #     self.assertTrue(np.allclose(_truth, truth[0, :, 0], atol=1e-7))

    # def test_mysimpledense(self):
    #     units = 1
    #     _init_kernel = 2
    #     _init_bias = 1
    #     input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
    #     mysimpledense = core.MySimpleDense(units=units, _init_kernel=_init_kernel, _init_bias=_init_bias, use_bias=True, debug=True)
    #     output_1 = mysimpledense(input_1)
    #     utils.init_tf_variables()
    #     result_1 = self.session.run(output_1)
    #     kernel = np.array([_init_kernel]*2).reshape(-1, 1)
    #     bias = np.array([_init_bias])
    #     truth = np.matmul(self.inputs.reshape([1, -1, 2]), kernel) + bias
    #     self.assertTrue(np.allclose(result_1, truth, atol=1e-7))

    # def test_average_layer(self):
    #     input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
    #     average = tf.keras.layers.Average()([input_1, 3*input_1])
    #     utils.init_tf_variables()
    #     result = self.session.run(average)
    #     self.assertTrue(np.allclose(result.reshape(-1), 2*self.inputs.reshape(-1), atol=1e-7))

    # def test_gradient_operator(self):
    #     input_1 = ops.convert_to_tensor(self.inputs.reshape([1, 1, -1]), dtype=tf.float32)
    #     operator = core.Operator(debug=False)
    #     output_1 = operator(input_1)
    #     gradient_by_tf = tf.gradients(output_1, input_1)[0]
    #     gradient_by_hand = core.gradient_operator(output_1, operator.kernel)
    #     Jacobian = core.jacobian(output_1, input_1)
    #     utils.init_tf_variables()
    #     result_by_tf, result_by_hand, J = self.session.run([gradient_by_tf, gradient_by_hand, Jacobian])

    #     self.assertTrue(np.allclose(np.diag(J).reshape(-1), result_by_hand.reshape(-1)))
    #     self.assertTrue(np.allclose(J.sum(axis=0).reshape(-1), result_by_tf.reshape(-1)))

    def test_gradient_mydense(self):
        # self._test_gradient_mydense_helper(activation=None)
        # self._test_gradient_mydense_helper(activation='tanh')
        # self._test_gradient_mydense_helper(activation='relu')
        # self._test_gradient_mydense_helper(activation='elu')
        # self._test_gradient_mydense_helper(activation='softmax')
        pass

    def test_gradient_mysimpledense(self):
        # input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 2]), dtype=tf.float32)
        # mysimpledense = core.MySimpleDense(units=1,
        #                                    use_bias=True,
        #                                    debug=False)
        # output_1 = mysimpledense(input_1)
        # gradient_by_tf = tf.gradients(output_1, input_1)[0]
        # utils.init_tf_variables()
        # gradient_by_hand = core.gradient_linear_layer(mysimpledense.kernel,
        #                                               multiples=self.inputs.shape[0]//input_1.shape[-1].value)

        # result_by_tf, result_by_hand = self.session.run([gradient_by_tf, gradient_by_hand])
        # self.assertTrue(np.allclose(result_by_tf, result_by_hand, atol=1e-7))
        pass

    def test_gradient_operator_mydense(self):
        # self._test_gradient_operator_mydense_helper(activation=None)
        # self._test_gradient_operator_mydense_helper(activation='tanh')
        # self._test_gradient_operator_mydense_helper(activation='relu')
        # self._test_gradient_operator_mydense_helper(activation='elu')
        # self._test_gradient_operator_mydense_helper(activation='softmax')
        pass

    def test_gradient_all(self):
        # self._test_gradient_all_helper(activation=None)
        # self._test_gradient_all_helper(activation='tanh')
        # self._test_gradient_all_helper(activation='relu')
        # self._test_gradient_all_helper(activation='elu')
        # self._test_gradient_all_helper(activation='softmax')
        pass

    def test_multiple_plays(self):
        # Note: test one by one, uncomment the case you want to test
        # self._test_multiple_plays_helper(1, None, 1)
        # self._test_multiple_plays_helper(1, None, 5)
        # self._test_multiple_plays_helper(2, None, 1)
        # self._test_multiple_plays_helper(2, None, 5)
        # self._test_multiple_plays_helper(1, 'tanh', 1)
        # self._test_multiple_plays_helper(1, 'tanh', 5)
        # self._test_multiple_plays_helper(2, 'tanh', 1)
        # self._test_multiple_plays_helper(2, 'tanh', 5)
        # self._test_multiple_plays_helper(1, 'relu', 1)
        # self._test_multiple_plays_helper(1, 'relu', 5)
        # self._test_multiple_plays_helper(2, 'relu', 1)
        # self._test_multiple_plays_helper(2, 'relu', 5)
        # self._test_multiple_plays_helper(1, 'elu', 1)
        # self._test_multiple_plays_helper(1, 'elu', 5)
        # self._test_multiple_plays_helper(2, 'elu', 1)
        # self._test_multiple_plays_helper(2, 'elu', 5)
        # self._test_multiple_plays_helper(2, 'softmax', 5)
        pass

    def _test_gradient_mydense_helper(self, activation):
        units = 10
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, -1, 1]), dtype=tf.float32)
        mydense = core.MyDense(units=units,
                               activation=activation,
                               use_bias=True,
                               debug=False)

        output_1 = mydense(input_1)
        gradient_by_tf = tf.gradients(output_1, input_1)[0]
        gradient_by_hand = core.gradient_nonlinear_layer(output_1, mydense.kernel, activation=activation)
        utils.init_tf_variables()
        result_by_tf, result_by_hand = self.session.run([gradient_by_tf, gradient_by_hand])
        self.assertTrue(np.allclose(result_by_tf.reshape(-1), result_by_hand.reshape(-1), atol=1e-7))

    def _test_gradient_operator_mydense_helper(self, activation):
        units = 5
        debug = False
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, 1, -1]), dtype=tf.float32)
        operator = core.Operator(debug=debug)
        output_1 = operator(input_1)
        mydense = core.MyDense(units=units,
                               activation=activation,
                               use_bias=True,
                               debug=debug)
        output_2 = mydense(output_1)
        gradient_by_tf = tf.gradients(output_2, input_1)[0]

        gradient_by_hand = core.gradient_operator_nonlinear_layers(output_1,
                                                                   output_2,
                                                                   operator.kernel,
                                                                   mydense.kernel,
                                                                   activation,
                                                                   debug=True,
                                                                   inputs=input_1)

        utils.init_tf_variables()
        result_by_tf, result_by_hand = self.session.run([gradient_by_tf, gradient_by_hand])
        self.assertTrue(np.allclose(result_by_tf.reshape(-1), result_by_hand.reshape(-1)))

    def _test_gradient_all_helper(self, activation):
        units = 5
        debug = False
        input_1 = ops.convert_to_tensor(self.inputs.reshape([1, 1, -1]), dtype=tf.float32)
        operator = core.Operator(debug=debug)
        output_1 = operator(input_1)
        mydense = core.MyDense(units=units,
                               activation=activation,
                               use_bias=True,
                               debug=debug)
        output_2 = mydense(output_1)

        mysimpledense = core.MySimpleDense(units=1,
                                           use_bias=True,
                                           activation=None,
                                           debug=debug)
        output_3 = mysimpledense(output_2)
        gradient_by_tf = tf.reshape(tf.gradients(output_3, input_1)[0], shape=output_1.shape)
        gradient_by_hand = core.gradient_all_layers(output_1,
                                                    output_2,
                                                    operator.kernel,
                                                    mydense.kernel,
                                                    mysimpledense.kernel,
                                                    activation,
                                                    debug=True,
                                                    inputs=input_1)
        utils.init_tf_variables()
        result_by_tf, result_by_hand = self.session.run([gradient_by_tf, gradient_by_hand])
        self.assertTrue(np.allclose(result_by_tf, result_by_hand))

    def _test_multiple_plays_helper(self, nb_plays, activation, input_dim):
        units = 5
        # timestep = self.inputs.shape[0] // input_dim
        timestep = 1
        mymodel = core.MyModel(nb_plays=nb_plays,
                               units=units,
                               input_dim=input_dim,
                               timestep=timestep,
                               activation=activation,
                               debug=True,
                               unittest=True)
        mymodel.compile(self.inputs[:input_dim], mu=0, sigma=1)
        utils.init_tf_variables()

        result_by_tf, result_by_hand, result_J_list_by_tf, result_J_list_by_hand = self.session.run([mymodel.J_by_tf, mymodel.J_by_hand,
                                                                                                     mymodel.J_list_by_tf, mymodel.J_list_by_hand],
                                                                                                    feed_dict=mymodel._x_feed_dict)
        for by_tf, by_hand in zip(result_J_list_by_tf, result_J_list_by_hand):
            if not np.allclose(by_hand, by_tf, atol=1e-7):
                print("ERROR: ")
                import ipdb; ipdb.set_trace()

        self.assertTrue(np.allclose(result_by_tf.reshape(-1), result_by_hand.reshape(-1), atol=1e-7))
        del mymodel


    def test_confusion_matrix(self):
        y_true = np.array([1, 2, 3, 1, 6, 3, 2, 7, 8, 9], dtype=np.float32)
        y_pred = np.array([2, 4, 1, 4, 6, 9, 3, 12, 2, 9], dtype=np.float32)
        confusion = core.confusion_matrix(y_true, y_pred)
        correct = np.array([[4, 3], [2, 0]], dtype=np.int32)
        self.assertTrue(np.all(confusion == correct))

    def test_simple_stateful(self):
        '''
        Given a state, compute the results corresponding to the state
        '''
        # FAILED
        # dim = 1
        # batch_size = 1

        # states = np.array([1]*batch_size).reshape((batch_size, dim))
        # input_1 = ops.convert_to_tensor(self.inputs.reshape([batch_size, 1, -1]), dtype=tf.float32)
        # operator_1 = core.Operator(debug=True)

        # output_1 = operator_1(input_1)
        # # init operator_1.state first, or it cannot assign by keras.set_value
        # operator_1.reset_states(states=None)
        # # utils.init_tf_variables()

        # # import ipdb; ipdb.set_trace()
        # operator_1.reset_states(states=states)
        # output_2 = operator_1(input_1)
        # utils.init_tf_variables()
        # operator_1.reset_states(states=states)
        # result_1 = self.session.run(output_1)

        # result_2, op_states = self.session.run([output_2, operator_1.states])
        # self.assertTrue(np.allclose(result_1.reshape(-1), self.truth_with_state_zero))
        # self.assertTrue(np.allclose(result_2.reshape(-1), self.truth_with_state_one))
        # self.assertTrue(np.allclose(states, op_states))

    # def test_stateful(self):
    #     input_dim = self.inputs.shape[-1]
    #     activation = None
    #     timestep = self.inputs.shape[0] // input_dim

    #     mymodel = core.MyModel(nb_plays=1,
    #                            units=5,
    #                            input_dim=input_dim,
    #                            timestep=timestep,
    #                            activation=activation,
    #                            debug=True)

    #     mymodel.compile(self.inputs, mu=0, sigma=1, test_stateful=True)
    #     ins = [self.inputs.reshape(1, 1, -1)]
    #     utils.init_tf_variables()

    #     states_list = [0] * mymodel._nb_plays
    #     mymodel.reset_states(states_list=states_list)
    #     op_output_1 = mymodel.train_function(ins)[0]

    #     self.assertTrue(np.allclose(op_output_1.reshape(-1), self.truth_with_state_zero))

    #     states_list = [1] * mymodel._nb_plays
    #     mymodel.reset_states(states_list=states_list)
    #     op_output_2 = mymodel.train_function(ins)[0]

    #     self.assertTrue(np.allclose(op_output_2.reshape(-1), self.truth_with_state_one))


    def test_stateful_model(self):
        # self._test_stateful_model_simple(nb_plays=1)
        # self._test_stateful_model_simple(nb_plays=2)

        # self._test_stateful_model(1)
        # self._test_stateful_model(2)

        # self._test_stateful_model(1, 2)
        # self._test_stateful_model(2, 2)

        # self._test_stateful_model(1, 5)
        # self._test_stateful_model(2, 5)

        # self._test_stateful_model(1, 20)
        # self._test_stateful_model(2, 20)
        pass

    def _test_stateful_model_simple(self, nb_plays):
        units = 5
        input_dim = 10          # it's batch_size

        activation = None
        input_1, input_2 = self.inputs[:10], self.inputs[10:]

        mymodel = core.MyModel(nb_plays=nb_plays,
                               units=units,
                               input_dim=input_dim,
                               timestep=1,
                               activation=activation,
                               debug=True)

        mymodel.compile(input_1, mu=0, sigma=1, test_stateful=True)

        ins = [input_1.reshape(mymodel.batch_input_shape)]
        utils.init_tf_variables()

        states_list = [0] * mymodel._nb_plays
        mymodel.reset_states(states_list=states_list)
        output_1 = mymodel.train_function(ins)

        ins = [input_2.reshape(mymodel.batch_input_shape)]
        states_list = [o.reshape(-1)[-1] for o in output_1]
        mymodel.reset_states(states_list=states_list)
        output_2 = mymodel.train_function(ins)

        results = []
        for o1, o2 in zip(output_1, output_2):
            result = np.hstack([o1.reshape(-1), o2.reshape(-1)])
            results.append(result)

        truth = [self.truth_with_state_zero] * mymodel._nb_plays
        for r, t in zip(results, truth):
            self.assertTrue(np.allclose(r, t))

    def _test_stateful_model(self, nb_plays, input_dim=10):
        length = self.inputs.shape[-1]
        units = 5
        activation = None
        steps_per_epoch = length // input_dim

        mymodel = core.MyModel(nb_plays=nb_plays,
                               units=units,
                               input_dim=input_dim,
                               timestep=1,
                               activation=activation,
                               debug=True)

        outputs = mymodel.fit2(self.inputs, mu=0, sigma=1, epochs=1, steps_per_epoch=steps_per_epoch, test_stateful=True)
        for o in outputs:
            self.assertTrue(np.allclose(o, self.truth))

    # def test_keras_set_value(self):
    #     v = tf.Variable(1.0)
    #     utils.init_tf_variables()
    #     tf.keras.backend.set_value(v, 0.0)
    #     # uncomment the following line, cause error
    #     # utils.init_tf_variables()
    #     r = self.session.run(v)
    #     self.assertEqual(r, 0)

    # def test_lstm(self):
    #     self._test_lstm_with_units(1, 1)
    #     self._test_lstm_with_units(1, 2)
    #     self._test_lstm_with_units(2, 1)
    #     self._test_lstm_with_units(2, 2)

    # def test_lstm_model(self):
    #     self._test_lstm_model_with_units(1, 1)
    #     self._test_lstm_model_with_units(1, 2)
    #     self._test_lstm_model_with_units(2, 1)
    #     self._test_lstm_model_with_units(2, 2)

    # def _test_lstm_with_units(self, units, batch_size):
    #     size = [1, 1, 64]
    #     x = np.random.normal(size=size)
    #     x = x.reshape(batch_size, -1, 1)
    #     lstm_layer = tf.keras.layers.LSTM(units, return_sequences=True)
    #     inputs = ops.convert_to_tensor(x, dtype=tf.float32)
    #     outputs = lstm_layer(inputs)
    #     self.assertTrue(outputs.shape.as_list() == [batch_size, 64 // batch_size, units])

    # def _test_lstm_model_with_units(self, units, batch_size):
    #     size = [1, 1, 64]
    #     x = np.random.normal(size=size).reshape(1, -1, 1)
    #     y = np.random.normal(size=size).reshape(1, -1, 1)
    #     inputs = ops.convert_to_tensor(x, dtype=tf.float32)
    #     outputs = ops.convert_to_tensor(y, dtype=tf.float32)

    #     model = tf.keras.models.Sequential()
    #     model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    #     model.compile(loss='mse', optimizer='adam')
    #     model.fit(inputs, outputs, verbose=1, epochs=1, steps_per_epoch=2)


    # def test_mean_of_J(self):
    #     a1 = tf.constant([1, 2, 3, 4], shape=(1, 4, 1))
    #     a2 = tf.constant([5, 6, 7, 8], shape=(1, 4, 1))
    #     a = tf.reduce_mean(tf.concat([a1, a2], axis=-1), axis=-1, keepdims=True)
    #     truth = tf.constant([3, 4, 5, 6], shape=(1, 4, 1))
    #     self.assertTrue(a.shape.as_list() == truth.shape.as_list())
    #     a_result, b_result = self.session.run([a, truth])
    #     self.assertTrue(np.allclose(a_result, b_result))

    # def test_is_tensor(self):
    #     a = np.array([1, 2, 3, 4])
    #     a1 = ops.convert_to_tensor(a)
    #     self.assertTrue(isinstance(a1, tf.Tensor))

    def test_activation_softmax(self):
        a = np.array([1, 2, 3, 4], dtype=np.float32).reshape([1, 4, 1])
        a1 = np.log(1 + np.exp(a))
        aa = tf.constant([1, 2, 3, 4], shape=(1, 4, 1), dtype=tf.float32)
        aa1 = core.my_softmax(aa)
        aa2 = self.session.run(aa1)
        self.assertTrue(np.allclose(a1, aa2))


    # def test_create_tensor(self):
    #     import time
    #     start = time.time()
    #     a = tf.placeholder(tf.float32, shape=None)
    #     b = tf.placeholder(tf.float32, shape=None)
    #     c = a * b
    #     for i in range(1000):
    #         self.session.run(c, feed_dict={a: 1, b: 2})
    #     end = time.time()
    #     print("cost time 1: {}s ".format(end-start))

    #     start = time.time()
    #     a = tf.constant([1], dtype=tf.float32)
    #     b = tf.constant([2], dtype=tf.float32)
    #     for i in range(1000):
    #         self.session.run(a*b)

    #     end = time.time()
    #     print("cost time 2: {}s ".format(end-start))

    #     start = time.time()
    #     a = tf.constant([1], dtype=tf.float32)
    #     b = tf.constant([2], dtype=tf.float32)
    #     for i in range(1000):
    #         c = a * b
    #         self.session.run(c)
    #     end = time.time()
    #     print("cost time 3: {}s ".format(end-start))

    # def test_tensordot(self):
    #     a = tf.constant([1, 2, 3, 4], shape=(1, 2, 2))
    #     b = tf.constant([2, 3], shape=(2, 1))
    #     c = tf.tensordot(a, b, axes=[[2], [0]])
    #     print(c.shape)
    #     print(self.session.run(c))
    #     d = tf.constant([1, 2, 1, 2], shape=[2, 2])
    #     e = tf.constant([1, 2, 2, 3], shape=[2, 2])
    #     f = d * e
    #     g = 1 - d
    #     print(self.session.run([f, g]))

    # def test_inputs(self):
    #     path = "new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv"
    #     import trading_data as tdata
    #     inputs, outputs = tdata.DatasetLoader.load_data(path)
    #     def phi(x):
    #         if x > 0.5:
    #             return x - 0.5
    #         elif x < -0.5:
    #             return x + 0.5
    #         else:
    #             return float(0)

    #     pn_list = [0]
    #     w = 2
    #     for x in inputs[:1500]:
    #         pn = phi(w * x - pn_list[-1]) + pn_list[-1]
    #         pn_list.append(pn)

    #     pn_list = np.array(pn_list)
    #     import ipdb; ipdb.set_trace()

    #     print("hello world")

    def test_extract_by_idx(self):
        embs = tf.constant(np.random.randint(0, 3, size=(3, 6, 16)), tf.int32)
        _ids = tf.constant([0, 1, 2, 0, 1, 2], shape=(1, 6, 1))
        multiple = tf.constant([3, 1, 16], tf.int32)
        ids = tf.tile(_ids, multiple)
        r1 = tf.cond(
            tf.reduce_any(tf.math.equal(x=embs, y=ids)),
            lambda: True,
            lambda: False
            # x=embs,
            # y=ids
        )
        import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    unittest.main()
