from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

from floodanddroughtpredictor.data_process import preprocess_univariate


def run_lstm_univariate(dataset, predicted_value):
    le = LabelEncoder()
    reframed, scaler = preprocess_univariate(dataset)
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

    pyplot.plot(inv_y, label = 'Actual {}'.format(predicted_value))
    pyplot.plot(inv_yhat, label = 'Predicted {}'.format(predicted_value))
    pyplot.legend()
    pyplot.show()
