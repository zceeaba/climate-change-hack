import calendar
import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM


def convert_timeseries_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    # covert timeseries data to t-n to t-1 form
    # n defines how many previous value should be taken into consideration
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis = 1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace = True)
    return agg


def run_lstm(dataset):
    le = LabelEncoder()
    dataset.columns = dataset.columns.str.lstrip()
    reverse_month = {month: index for index, month in enumerate(calendar.month_abbr) if month}
    dataset['Statistics'] = dataset['Statistics'].apply(
        lambda x: reverse_month[x.split()[0]]
    )
    dataset['time_period'] = dataset.apply(lambda x: datetime.date(int(x[1]), int(x[2]), 1), axis=1)
    dataset = dataset.drop(columns=['Year', 'Statistics', 'Country', 'ISO3'])
    dataset = dataset.sort_values('time_period')
    dataset.set_index('time_period', inplace = True)
    data = dataset.values
    # ensure all data is float
    values = data.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = convert_timeseries_to_supervised(scaled, 1, 1)

    print(reframed.head())
    # drop columns we don't want to predict
    # need to change this if we change N or change dataset
    # reframed.drop(reframed.columns[[4, 5]], axis = 1, inplace = True)
    # print(reframed.head())

    # split into train and test sets
    values = reframed.values
    train_X, test_X, train_y, test_y = train_test_split(
        values[:, :-1], values[:, -1], test_size = 0.2
    )

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape = (train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss = 'mae', optimizer = 'adam')

    # fit network
    history = model.fit(
        train_X, train_y, epochs = 50, batch_size = 72,
        validation_data = (test_X, test_y), verbose = 2,
        shuffle = False
    )
    # plot history
    pyplot.plot(history.history['loss'], label = 'Training Loss')
    pyplot.plot(history.history['val_loss'], label = 'Validation Loss')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast to revert data into original form
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis = 1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    # Actial Input
    inv_xp = inv_yhat[:, 1:]
    # predicted output
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis = 1)
    inv_y = scaler.inverse_transform(inv_y)
    # Actual output
    inv_y = inv_y[:, 0]
    print("Actial Input:")
    print(inv_xp)
    print("Actual output:")
    print(inv_y)
    # predicted output will be offset by 1
    print("Predicted output:")
    print(inv_yhat)

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    pyplot.plot(inv_y, label = 'Actual Sales')
    pyplot.plot(inv_yhat, label = 'Predicted Sales')
    pyplot.legend()
    pyplot.show()
