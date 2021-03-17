import pandas as pd

from floodanddroughtpredictor.data_import import load_data
from floodanddroughtpredictor.data_analysis import run_lstm

if __name__ == '__main__':
    rain_data = load_data('pr_1991_2016_NER.csv')
    temp_data = load_data('tas_1991_2016_NER.csv')
    run_lstm(temp_data, 'temperature')
    run_lstm(rain_data, 'rain')
