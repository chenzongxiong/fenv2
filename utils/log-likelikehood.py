import sys

sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..")

import trading_data as tdata

if __name__ == '__main__':
    sigma = 20
    ensemble = 3
    base_fname = './new-dataset/models/diff_weights/method-stock/activation-None/state-0/mu-0/sigma-{sigma}/units-0/nb_plays-0/points-0/input_dim-1/base.csv'.format(sigma=sigma)
    inputs, outputs = tdata.DatasetLoader.load_data(base_fname)
    inputs, outputs = inputs[:1700], outputs[:1700]

    _test_inputs, _test_outputs = inputs[1300:], outputs[1300:]


    predict_fname = './new-dataset/lstm/diff_weights/method-stock/activation-None/state-0/input_dim-1/mu-0/sigma-{sigma}/units-0/nb_plays-0/points-0/units#-64/ensemble-{ensemble}/loss-mle/predictions.csv'.format(sigma=sigma,                                                                                                                                                                                                          ensemble=ensemble)
    _predict_inputs, _predict_outputs = tdata.DatasetLoader.load_data(predict_fname)
    # import ipdb; ipdb.set_trace()
    _test_diff_outputs = _test_outputs[1:] - _test_outputs[:-1]
    _predict_diff_outputs = _predict_outputs[1:] - _predict_outputs[:-1]
    rmse1 = ((_predict_diff_outputs - _test_diff_outputs)**2).mean() ** 0.5
    rmse2 = ((_predict_diff_outputs + _test_diff_outputs)**2).mean() ** 0.5
    print("rmse1: ", rmse1, ", rmse2: ", rmse2)

    predict_mu = _predict_diff_outputs.mean()
    predict_std = _predict_diff_outputs.std()
    ground_truth_mu = _test_diff_outputs.mean()
    ground_truth_std = _test_diff_outputs.std()

    print('mu: ', ground_truth_mu, ', std: ', ground_truth_std)
    print('predict_mu: ', predict_mu, ', predict_std: ', predict_std)
