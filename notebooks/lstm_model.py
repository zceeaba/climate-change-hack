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

def visualize_loss(history, title, filename):
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
	plt.savefig('data/loss/'+str(filename)+'.png')

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
dataset = pd.read_csv('/Users/henrietta.ridley/climate_data/data/spei01_niger.csv', header=0, infer_datetime_format=True, parse_dates=['time'], index_col=['time'])
dataset['id'] = dataset['lon'] + dataset['lat']
dataset_notnull = dataset.dropna()

test = dataset_notnull.id.unique()[:25]

output = pd.DataFrame()
for id in test:
	dataset_subset = dataset_notnull.loc[dataset_notnull['id'] == id]
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
	verbose, epochs, batch_size, learning_rate = 1, 100, 20, 0.0001
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

	visualize_loss(history, "Training and Validation Loss", id)

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
	# frame = pd.concat([
	#  pd.DataFrame({'time': months, '(t+1)':actuals_1, '(t+2)':actuals_2, '(t+3)':actuals_3, 'type': 'original'}),
	#  pd.DataFrame({'time': months, '(t+1)':preds_1, '(t+2)':preds_2, '(t+2)':preds_3, 'type': 'forecast'})])

	df1 = pd.DataFrame({'time': months, 'original (t+1)':actuals_1, 'original (t+2)':actuals_2, 'original (t+3)':actuals_3})
	df2 = pd.DataFrame({'time': months, 'forecast (t+1)':preds_1, 'forecast (t+2)':preds_2, 'forecast (t+2)':preds_3})

	frame = pd.merge(df1, df2, on="time")
	frame['id'] = id
	output = output.append(frame)

# Join the forecasting data onto the geo dataframe
output_df = pd.merge(dataset_notnull, output, how = 'inner', on = ('time', 'id'))
output_df.columns

# Attach and save this down as an
import geopandas as gpd
import pandas as pd

geom = [Point(x,y) for x, y in zip(output_df['lon'], output_df['lat'])]
niger_drought = gpd.GeoDataFrame(output_df, geometry=geom)

niger_drought.to_file("/Users/henrietta.ridley/climate_data/data/niger_drought_preds/niger_drought.shp")
