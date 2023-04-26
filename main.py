#Teste Optuna Corn dataset.
#Implement collecting Scores in Cross Validation


import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

import data_prep
import model_tunning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")


def data_build():
    target = 'corn_price'

    data, df_calendar = data_prep.open_dataset()
    #df = data_prep.clean_dummies(data)
    df = data
    forecast_horizon = 1
    date_col = 'data'
    last_date = '202212'
    data_train, data_test = data_prep.make_lags(df, df_calendar, forecast_horizon, last_date, date_col, target)

    print('Using only train dataset, leaving test_set for later (3 last Observations)')


    x_train = data_train.loc[:, data_train.columns != target].set_index('data')
    y_train = data_train[[target, 'data']].set_index('data')

    x_test = data_test.loc[:, data_test.columns != target].set_index('data')
    y_test = data_test[[target, 'data']].set_index('data')

    return x_train, y_train, x_test, y_test





if __name__ == '__main__':

    print('Load X and y Train and Test')
    x_train, y_train, x_test, y_test = data_build()
    #Lags created and past info were deslocated into future. That is not data leakeage, since none data from test set
    #was included into train set
    print('Checking split between train and validation sets based on date index')
    tss_cv_ = TimeSeriesSplit(n_splits=5)

    for train_index, valid_index in tss_cv_.split(x_train):
        train, valid = x_train.iloc[train_index], x_train.iloc[valid_index]

    print('Dictionary with best param for defined models and optuna objects')
    #Tunning with Optuna, using only train dataset
    dict_hyperparams, dict_studies = model_tunning.models_tuning(x_train, y_train)
    #Predict and check results
    print('Re use best trail that achieved best hyperparams and predict test set')

    print('Even though we have the studies, the trained was made on partial data. Now will train full train dateset')

    df_predictions, scores_teste = model_tunning.training_full_data


