import calendar
import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler


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


def preprocess_univariate(dataset):
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

    return reframed, scaler

