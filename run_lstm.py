# https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm
import sys
import argparse
import time

import trading_data as tdata
import tensorflow as tf
import numpy as np


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)

# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# units = [1500]
# capcity  = [1, 2, 4, 8, 16, 32]
def lstm_return_sequence(input_fname, units, capacity=1, epochs=5000):
    start = time.time()
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)
    _timestamp = np.arange(2000)
    units = 1500
    prices = _prices[:units]
    timestamp = _timestamp[:units].reshape(-1, 1, 1)
    prices = prices.reshape(-1, 1, 1)   # (batch_size, timestamp, input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(capacity,
                                   batch_size=1,
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   stateful=True,
                                   implementation=2))

    model.add(tf.keras.layers.Dense(units=1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    model.fit(timestamp, prices, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback], validation_split=0.05,
              shuffle=False)

    pred_inputs = np.hstack([_timestamp[units:2000], _timestamp[:1000]])
    truth_outputs = np.hstack([_prices[units:2000], _prices[:1000]])

    pred_inputs = _timestamp[units:2000]
    truth_outputs = _prices[units:2000]

    pred_inputs = pred_inputs.reshape(-1, 1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs[:100] - pred_outputs[:100]) ** 2)

    print("LSTM mse: {}".format(mse))

    output_fname = "new-dataset/lstm/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=capacity)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)

    end = time.time()
    print("time cost: {}".format(end- start))


def lstm_without_sequence(inputs_fname, units, epochs=5000):
    start = time.time()
    units = 1500
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)
    _timestamp = np.arange(2000)
    prices = _prices[:units]
    timestamp = _timestamp[:units]
    timestamp = timestamp[:units].reshape(-1, 1, 1)
    prices = prices.reshape(-1, 1)   # (batch_size, timestamp, input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(capacity,
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=False,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    model.fit(timestamp, prices, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback], validation_split=0.05,
              shuffle=False)

    # pred_inputs = np.hstack([_timestamp[units:2000], _timestamp[:1000]])
    # truth_outputs = np.hstack([_prices[units:2000], _prices[:1000]])

    pred_inputs = _timestamp[units:2000]
    truth_outputs = _prices[units:2000]

    pred_inputs = pred_inputs.reshape(-1, 1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs[:100] - pred_outputs[:100]) ** 2)
    print("LSTM mse: {}".format(mse))

    output_fname = "new-dataset/lstm/lstm2/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=capacity)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)

    end = time.time()
    print("time cost: {}".format(end- start))


# price vs. price
def lstm_3(input_fname, units, capacity=1, epochs=5000):

    start = time.time()
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)
    units = 1500

    prices1 = _prices[:units]
    prices2 = _prices[1:units+1]

    prices1 = prices1.reshape(-1, 1, 1)   # (batch_size, timestamp, input_dim)
    prices2 = prices2.reshape(-1, 1, 1)   # (batch_size, timestamp, input_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(capacity),
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    model.fit(prices1, prices2, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback], validation_split=0.05,
              shuffle=False)

    # pred_inputs = np.hstack([_prices[units:2000], _prices[:1000]])
    # truth_outputs = np.hstack([_prices[units+1:2001], _prices[1:1001]])

    pred_inputs = _prices[units:2000]
    truth_outputs = _prices[units+1:2001]

    pred_inputs = pred_inputs.reshape(-1, 1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs[:100] - pred_outputs[:100]) ** 2)

    print("LSTM mse: {}".format(mse))
    output_fname = "new-dataset/lstm/price_vs_price/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=capacity)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)

    end = time.time()
    print("time cost: {}".format(end- start))


def lstm_4(input_fname, units, capacity=1, epochs=5000):
    start = time.time()
    units = 1500
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)

    prices1 = _prices[:units]
    prices2 = _prices[1:units+1]

    prices1 = prices1.reshape(-1, 1, 1)   # (batch_size, timestamp, input_dim)
    prices2 = prices2.reshape(-1, 1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(capacity,
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=False,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
    model.summary()
    model.fit(prices1, prices2, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback], validation_split=0.05,
              shuffle=False)


    pred_inputs = _prices[1500:2000]
    truth_outputs = _prices[1501:2001]

    pred_inputs = pred_inputs.reshape(-1, 1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs[:100] - pred_outputs[:100]) ** 2)
    print("LSTM mse: {}".format(mse))

    output_fname = "new-dataset/lstm/lstm4/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=capacity)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)

    end = time.time()
    print("time cost: {}".format(end- start))


def lstm_5(input_fname, units, capacity=1, epochs=5000):
    # https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm
    start = time.time()
    units = 1500
    _prices, _ = tdata.DatasetLoader.load_data(input_fname)

    prices1 = _prices[:units]
    prices2 = _prices[1:units+1]

    prices1 = prices1.reshape(-1, 1, 1)   # (batch_size, timestamp, input_dim)
    prices2 = prices2.reshape(-1, 1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mse'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=500)
    model = tf.keras.models.Sequential()

    capacity1 = capacity
    capacity2 = 2 * capacity1

    model.add(tf.keras.layers.LSTM(capacity1,
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=True,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(capacity2,
                                   input_shape=(1, 1),
                                   unroll=False,
                                   return_sequences=False,
                                   use_bias=True,
                                   stateful=True,
                                   batch_size=1,
                                   implementation=2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])

    model.summary()
    model.fit(prices1, prices2, epochs=epochs, verbose=1,
              callbacks=[early_stopping_callback], validation_split=0.05,
              shuffle=False)

    pred_inputs = _prices[1500:2000]
    truth_outputs = _prices[1501:2001]

    pred_inputs = pred_inputs.reshape(-1, 1, 1)
    predictions = model.predict(pred_inputs)
    pred_outputs = predictions.reshape(-1)

    mse = np.mean((truth_outputs[:100] - pred_outputs[:100]) ** 2)
    print("LSTM mse: {}".format(mse))

    output_fname = "new-dataset/lstm/lstm5/price_vs_price/units-{units}/capacity-{capacity}/predictions.csv".format(units=units, capacity=capacity)
    tdata.DatasetSaver.save_data(pred_inputs.reshape(-1), pred_outputs.reshape(-1), output_fname)

    end = time.time()
    print("time cost: {}".format(end- start))




if __name__ == "__main__":

    input_fname = "new-dataset/models/diff_weights/method-sin/activation-None/state-0/markov_chain/mu-0/sigma-110/units-10000/nb_plays-20/points-1000/input_dim-1/mu-0-sigma-110-points-1000.csv"


    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", dest="capacity",
                        required=False, default=1,
                        type=int)

    parser.add_argument("--epochs", dest="epochs",
                        required=False, default=100,
                        type=int)

    parser.add_argument("--method", dest="method",
                        required=False, default=1,
                        type=int)

    argv = parser.parse_args(sys.argv[1:])
    method = argv.method
    units = 1500
    epochs = argv.epochs
    capacity = argv.capacity

    print("====================INFO====================")
    print("units: {}".format(units))
    print("method: {}".format(method))
    print("epochs: {}".format(epochs))
    print("capacity: {}".format(capacity))
    print("================================================================================")

    if method == 1:
        lstm_return_sequence(input_fname, units, capacity=capacity, epochs=epochs)
    elif method == 2:
        lstm_without_sequence(input_fname, units, capacity=capacity, epochs=epochs)
    elif method == 3:
        lstm_3(input_fname, units, capacity=capacity, epochs=epochs)
    elif method == 4:
        lstm_4(input_fname, units, capacity=capacity, epochs=epochs)
    elif method == 5:
        lstm_5(input_fname, units, capacity=capacity, epochs=epochs)
    else:
        pass
