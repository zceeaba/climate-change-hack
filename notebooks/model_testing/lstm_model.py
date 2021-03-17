# univariate multi-step lstm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import keras

# # # split a univariate dataset into train/test sets
# # def split_dataset(data):
# #     # split into standard months
# #     train, test = data[1:-120], data[-120:-5]
# #     # restructure into windows of 6 monthly data
# #     train = array(split(train, len(train )/6))
# #     test = array(split(test, len(test )/6))
# #     return train, test
#
# # evaluate one or more weekly forecasts against expected values
# def evaluate_forecasts(actual, predicted):
#     scores = list()
#     # calculate an RMSE score for each day
#     for i in range(actual.shape[1]):
#         # calculate mse
#         mse = mean_squared_error(actual[:, i], predicted[:, i])
#         # calculate rmse
#         rmse = sqrt(mse)
#         # store
#         scores.append(rmse)
#     # calculate overall RMSE
#     s = 0
#     for row in range(actual.shape[0]):
#         for col in range(actual.shape[1]):
#             s += (actual[row, col] - predicted[row, col] )**2
#     score = sqrt(s / (actual.shape[0] * actual.shape[1]))
#     return score, scores
#
# # summarize scores
# def summarize_scores(name, score, scores):
#     s_scores = ', '.join(['%.1f' % s for s in scores])
#     print('%s: [%.3f] %s' % (name, score, s_scores))
# #
# # # convert history into inputs and outputs
# # def to_supervised(train, n_input, n_out=3):
# #     # flatten data
# #     data = train.reshape((train.shape[0 ] *train.shape[1], train.shape[2]))
# #     X, y = list(), list()
# #     in_start = 0
# #     # step over the entire history one time step at a time
# #     for _ in range(len(data)):
# #         # define the end of the input sequence
# #         in_end = in_start + n_input
# #         out_end = in_end + n_out
# #         # ensure we have enough data for this instance
# #         if out_end <= len(data):
# #             x_input = data[in_start:in_end, 0]
# #             x_input = x_input.reshape((len(x_input), 1))
# #             X.append(x_input)
# #             y.append(data[in_end:out_end, 0])
# #         # move along one time step
# #         in_start += 1
# #     return array(X), array(y)
# #
# # # train the model
# # def build_model(train, n_input):
# #     # prepare data
# #     train_x, train_y = to_supervised(train, n_input)
# #     # define parameters
# #     verbose, epochs, batch_size = 0, 70, 16
# #     n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
# #     # define model
# #     model = Sequential()
# #     model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
# #     model.add(Dense(100, activation='relu'))
# #     model.add(Dense(n_outputs))
# #     model.compile(loss='mse', optimizer='adam')
# #     # fit network
# #     model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
# #     return model
#
# # make a forecast
# def forecast(model, history, n_input):
#     # flatten data
#     data = array(history)
#     data = data.reshape((data.shape[0 ] *data.shape[1], data.shape[2]))
#     # retrieve last observations for input data
#     input_x = data[-n_input:, 0]
#     # reshape into [1, n_input, 1]
#     input_x = input_x.reshape((1, len(input_x), 1))
#     # forecast the next week
#     yhat = model.predict(input_x, verbose=0)
#     # we only want the vector forecast
#     yhat = yhat[0]
#     return yhat
#
# # evaluate a single model
# def evaluate_model(train, test, n_input):
#     # fit model
#     model = build_model(train, n_input)
#     # history is a list of weekly data
#     history = [x for x in train]
#     # walk-forward validation over each week
#     predictions = list()
#     for i in range(len(test)):
#         # predict the week
#         yhat_sequence = forecast(model, history, n_input)
#         # store the predictions
#         predictions.append(yhat_sequence)
#         # get real observation and add to history for predicting the next week
#         history.append(test[i, :])
#     # evaluate predictions days for each week
#     predictions = array(predictions)
#     score, scores = evaluate_forecasts(test[:, :, 0], predictions)
#     return score, scores

# normalise the data
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg

def create_x_y(data, n_input, n_out):
    supervised = series_to_supervised(data, n_input, n_out)
    supervised_values = supervised.values
    # reshape training into [samples, timesteps, features]
    x, y = supervised_values[:, 0:n_input], supervised_values[:, n_input:]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    return x, y, supervised_values

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

def Extract(lst, idx):
    return [item[idx] for item in lst]

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# load the new file and subset so that we only have a single point
dataset = pd.read_csv('/data/spei01_niger.csv', header=0, infer_datetime_format=True, parse_dates=['time'], index_col=['time'])
dataset['id'] = dataset['lon'] + dataset['lat']
dataset_notnull = dataset.dropna()
dataset_notnull.id[2]

dataset_subset = dataset_notnull.loc[dataset_notnull['id'] == 27.850000381469727]
dataset_subset = dataset_subset['spei']

# split into train and test
split_fraction = 0.715
train_split = int(split_fraction * int(dataset_subset.shape[0]))

# Normalise the data
spei = normalize(dataset_subset.values, train_split)
data = pd.DataFrame(spei)
data.head()

train_data = data.loc[0: train_split - 1]
val_data = data.loc[train_split:]

# Convert time series to supervised problem
n_input = 12 # how many records back are we looking at to create each prediction
n_out = 3 # how many time steps forward do we want to predict for

train_x, train_y, train = create_x_y(train_data, n_input, n_out)
val_x, val_y, val = create_x_y(val_data, n_input, n_out)

# define parameters
verbose, epochs, batch_size, learning_rate = 1, 70, 16, 0.0001
n_features, n_timesteps, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_features, n_timesteps)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
model.summary()

# fit network
# model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
history = model.fit(train_x, train_y
                    , epochs=epochs
                    , batch_size=batch_size
                    , verbose=verbose
                    , validation_data=(val_x, val_y))

visualize_loss(history, "Training and Validation Loss")


# make forecasts
forecasts = make_forecasts(model, batch_size, train, val, n_timesteps, n_outputs)

actuals = val_y
preds = np.asarray(forecasts, dtype=np.float32)

actuals_1, actuals_2, actuals_3 = Extract(actuals,0), Extract(actuals,1), Extract(actuals,2)
preds_1, preds_2, preds_3 = Extract(preds,0), Extract(preds,1), Extract(preds,2)

# Comparing the forecasts with the actual values

# Creating the frame to store both predictions
out = dataset_subset.reset_index()
months = out['time'].values[-len(actuals):]
frame = pd.concat([
 pd.DataFrame({'month': months, '(t+1)':actuals_1, '(t+2)':actuals_2, '(t+3)':actuals_3, 'type': 'original'}),
 pd.DataFrame({'month': months, '(t+1)':preds_1, '(t+2)':preds_2, '(t+2)':preds_3, 'type': 'forecast'})])


# inverse transform forecasts and test
actual = [row[n_timesteps:] for row in val]
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_timesteps, n_outputs)




# # invert differenced forecast
# def inverse_difference(last_ob, forecast):
# 	# invert first forecast
# 	inverted = list()
# 	inverted.append(forecast[0] + last_ob)
# 	# propagate difference forecast using inverted first value
# 	for i in range(1, len(forecast)):
# 		inverted.append(forecast[i] + inverted[i-1])
# 	return inverted

# inverse data transform on forecasts
# def inverse_transform(series, forecasts, scaler, n_test):
# 	inverted = list()
# 	for i in range(len(forecasts)):
# 		# create array from forecast
# 		forecast = array(forecasts[i])
# 		forecast = forecast.reshape(1, len(forecast))
# 		# invert scaling
# 		inv_scale = scaler.inverse_transform(forecast)
# 		inv_scale = inv_scale[0, :]
# 		# invert differencing
# 		index = len(series) - n_test + i - 1
# 		last_ob = series.values[index]
# 		inv_diff = inverse_difference(last_ob, inv_scale)
# 		# store
# 		inverted.append(inv_diff)
# 	return inverted

# # make a forecast
# def forecast(model, history, n_input):
#     # flatten data
#     data = array(history)
#     data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
#     # retrieve last observations for input data
#     input_x = data[-n_input:, 0]
#     # reshape into [1, n_input, 1]
#     input_x = input_x.reshape((1, 1, len(input_x)))
#     # forecast the next week
#     yhat = model.predict(input_x, verbose=0)
#     # we only want the vector forecast
#     yhat = yhat[0]
#     return yhat
#
#
# # evaluate a single model
# def evaluate_model(train, test, n_input):
#     # fit model
#     model = build_model(train, n_input)
#     # history is a list of weekly data
#     history = [x for x in train_x]
#     # walk-forward validation over each week
#     predictions = list()
#     for i in range(len(val_data)):
#         # predict the week
#         yhat_sequence = forecast(model, history, n_input)
#         # store the predictions
#         predictions.append(yhat_sequence)
#         # get real observation and add to history for predicting the next week
#         history.append(val_data[i, :])
#     # evaluate predictions days for each week
#     predictions = array(predictions)
#     score, scores = evaluate_forecasts(test[:, :, 0], predictions)
#     return score, scores
#
#
#
#
#
#
# def show_plot(plot_data, delta, title):
#     labels = ["History", "True Future", "Model Prediction"]
#     marker = [".-", "rx", "go"]
#     time_steps = list(range(-(plot_data[0].shape[0]), 0))
#     if delta:
#         future = delta
#     else:
#         future = 0
#
#     plt.title(title)
#     for i, val in enumerate(plot_data):
#         if i:
#             plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
#         else:
#             plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
#     plt.legend()
#     plt.xlim([time_steps[0], (future + 5) * 2])
#     plt.xlabel("Time-Step")
#     plt.show()
#     return
#
#
# for x, y in train_x[:5], train_y[:5]:
#     show_plot(
#         [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
#         12,
#         "Single Step Prediction",
#     )
#
# # # define model
# # model = Sequential()
# # model.add(LSTM(32, input_shape=(train_x.shape)))
# # model.add(Dense(n_outputs))
# # model.compile(loss='mean_squared_error', optimizer='adam')
# # model.summary()
# #
# #
# # # inputs = keras.layers.Input(shape=(train_x.shape[1], train_x.shape[2]))
# # # lstm_out = keras.layers.LSTM(32)(inputs)
# # # outputs = keras.layers.Dense(1)(lstm_out)
# # # model = keras.Model(inputs=inputs, outputs=outputs)
# # # model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
# # #
# # #
# # #
# # # model = Sequential()
# # # model.add(LSTM(200, activation='relu', input_shape=inputs.shape[1], inputs.shape[2]))
# # # model.add(Dense(100, activation='relu'))
# # # model.add(Dense(n_outputs))
# # # model.compile(loss='mse', optimizer='adam')
# # # fit network
# # model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
# #
# #
#
#
#
#
#
#
#
# # design network
# model = Sequential()
# model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
# model.add(Dense(y.shape[1]))
# model.compile(loss='mean_squared_error', optimizer='adam')
# # fit network
# for i in range(nb_epoch):
#     model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
#     model.reset_states()
#
# # prepare data
# train_x, train_y = to_supervised(train, n_input, n_out)
#
#


# # evaluate model and get scores
# n_input = 12
# score, scores = evaluate_model(train, test, n_input)
# # summarize scores
# summarize_scores('lstm', score, scores)
# # plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
# pyplot.plot(days, scores, marker='o', label='lstm')
# pyplot.show()