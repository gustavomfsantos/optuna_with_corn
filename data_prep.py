import numpy as np
import pandas as pd

import warnings
from sklearn.exceptions import DataConversionWarning
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime as dt
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")


def open_dataset():
    df = pd.read_csv(r"C:\Users\gusta\Desktop\Math Projects\Price\corn_price_br\data\interim\corn_features.csv")
    print('Dataset features loaded')
    df_calendar = pd.read_csv(r"C:\Users\gusta\Desktop\Math "
                              r"Projects\Price\corn_price_br\data\interim\calendar_features.csv")
    print('Dataset calendar loaded')
    df['data'] = (df['data'].astype(str).str[:4] + df['data'].astype(str).str[5:7]).astype(int)

    df_calendar['data'] = (df_calendar['data'].astype(str).str[:4] + df_calendar['data'].astype(str).str[5:7]).astype(int)
    return df, df_calendar

def clean_dummies(data):
    dummies_many_zeros = list(data.columns[data.eq(0).mean()>0.90])
    print('Retirar da lista dummies de calendario')
    dummies_many_zeros_ = [x for x in dummies_many_zeros if 'd_20' not in x]
    data_new = data.drop(dummies_many_zeros_, axis=1)
    data_new = data_new.sort_values('data', ascending=True)
    data_new = data_new[data_new['data'] >= 200601]
    data_new = data_new.drop(['ACSP12_SCPCC12', 'ACSP12_TELCH12', 'estoque em t'], axis=1)
    print('New data without some dummies with too much zeros')
    return data_new

def isolate_test_data(data_new):
    print('Separate between Train and Test')
    print('Test should be last 3 observations')
    data_test = data_new.tail(3)
    data_test = data_test.iloc[:, :]
    data_train = data_new.iloc[:-3, :]
    print('Last columns with too many NAN removed')
    return data_train, data_test


def make_lags(df, df_calendar, forecast_horizon, last_date, date_col, target):

    last_date = int(last_date)

    index_aux = df_calendar[df_calendar['data'] == last_date].index.item() + forecast_horizon
    calendar_predict = df_calendar[df_calendar.index == index_aux]
    date_pred = calendar_predict['data'].item()

    if date_pred not in df['data'].to_list():
        df = df.append(calendar_predict)
    calendar_variables = ['feriado_nacional', 'dias_uteis', 'dias_tot',
                          'prop_dias_uteis', 'verao', 'outono', 'primavera',
                          'inverno', 'corona']

    default_variables = ['data',  ]

    features_variables = sorted(
        list(set(df.columns) - set(default_variables + calendar_variables )))

    lag_price = [1, 2, 3]
    lag_features = [1, 2]

    df_ = df.copy()
    df_[f'{target}_primeira_diff'] = df_[target].diff()
    df_[f'{target}_segunda_diff'] = df_[target].diff(2)

    gt_cols = [x for x in df_.columns if 'gt' in x]
    df_[gt_cols] = df_[gt_cols].where(df_[gt_cols].values == 0,
                                      other=(df_[gt_cols].fillna(method='ffill') + \
                                             df_[gt_cols].fillna(method='bfill')) / 2)

    df_filter = df_[default_variables + [target] + calendar_variables]

    for lag in lag_price:
        for col in [x for x in df_.columns if 'corn_price' in x]:
            if col in df_.columns:
                df_filter[col + '_lag' + str(lag + forecast_horizon - 1)] = df_[col].shift(
                    round(lag + forecast_horizon - 1))

    for lag in lag_features:
        for col in features_variables:
            if col in df_.columns:
                df_filter[col + '_lag' + str(lag + forecast_horizon - 1)] = df_[col].shift(
                    round(lag + forecast_horizon - 1))

    target_lags = sorted([x for x in df_filter.columns if target in x and 'lag' in x])
    target_lag = target_lags[0]


    df_filter = df_filter.assign(rol_sum3=df_filter.rolling(3, min_periods=1)[target_lag].mean())
    df_filter = df_filter.assign(rol_mean3=df_filter.rolling(3, min_periods=1)[target_lag].mean())
    df_filter = df_filter.assign(rol_mean6=df_filter.rolling(6, min_periods=1)[target_lag].mean())
    # df_filter = df_filter.assign(rol_mean9 = df_filter.rolling(9, min_periods=1)[target_lag].mean())
    df_filter = df_filter.assign(rol_mean12 = df_filter.rolling(12, min_periods=1)[target_lag].mean())
    # = df_filter.assign(rol_median3=df_filter.rolling(3, min_periods=1)[target_lag].median())
    #df_filter = df_filter.assign(rol_median6=df_filter.rolling(6, min_periods=1)[target_lag].median())
    # df_filter = df_filter.assign(rol_median9 = df_filter.rolling(9, min_periods=1)[target_lag].median())
    # df_filter = df_filter.assign(rol_median12 = df_filter.rolling(12, min_periods=1)[target_lag].median())
    df_filter = df_filter.assign(rol_std3=df_filter.rolling(3, min_periods=1)[target_lag].std())
    df_filter = df_filter.assign(rol_std6=df_filter.rolling(6, min_periods=1)[target_lag].std())
    # df_filter = df_filter.assign(rol_std9 = df_filter.rolling(9, min_periods=1)[target_lag].std())
    df_filter = df_filter.assign(rol_std12 = df_filter.rolling(12, min_periods=1)[target_lag].std())

    for i, col in enumerate(target_lags[0:3]):
        df_filter[f'mean_div{i}'] = df_filter['rol_mean3'] - df_filter[col]
        df_filter[f'sum_div{i}'] = df_filter['rol_sum3'] - df_filter[col]
        #df_filter[f'median_div{i}'] = df_filter['rol_median3'] - df_filter[col]

        df_filter[f'mean_div{i}_tag'] = 0
        df_filter.loc[df_filter[f'mean_div{i}'] > 0, f'mean_div{i}_tag'] = 1
        df_filter[f'median_div{i}_tag'] = 0
        #df_filter.loc[df_filter[f'median_div{i}'] > 0, f'median_div{i}_tag'] = 1



    for lag_col in [x for x in target_lags if 'lag1' in x]:
        lag = lag_col.split("_")[-1]

        for lag_ in [1, 3, 6, 9, 12]:
            try:
                df_filter[f'trend_{lag}_{lag_}'] = seasonal_decompose(df_filter[lag_col].dropna(), model='additive',
                                                                      period=lag_).trend
                df_filter[f'seas_{lag}_{lag_}'] = seasonal_decompose(df_filter[lag_col].dropna(),
                                                                     period=lag_).seasonal.fillna(
                    method="ffill").fillna(method="bfill")
                df_filter[f'resid_{lag}_{lag_}'] = seasonal_decompose(df_filter[lag_col].dropna(),
                                                                      period=lag_).resid.fillna(method="ffill").fillna(
                    method="bfill")
            except:
                print(f"Dataframe to small for {lag} and {lag_}")
    seas_cols = [x for x in df_filter.columns if 'seas' in x]


    print("Applying log in big coluns...")
    log_cols = df_filter.set_index('data').median().to_frame().reset_index()
    log_cols.columns = ['name', 'value']
    log_cols = log_cols[~(log_cols['name'].isin(default_variables))]
    log_cols = log_cols[~(log_cols['name'].isin(calendar_variables))]
    log_cols = log_cols[log_cols['value'] > 100000]['name'].to_list()
    # log_cols = [x for x in log_cols if "ordem"  not in x]
    log_cols = [x for x in log_cols if target[0] not in x]

    # print("Applying log in: ", log_cols)
    for col in log_cols:
        # print(col)
        df_filter[col] = df_filter[col].replace(0, 1)
        df_filter[col] = np.log10(df_filter[col])



    years = df_filter[date_col].apply(lambda x: str(x)[:4]).unique()
    for year in years:
        df_filter['year_{}'.format(year)] = 0
        df_filter.loc[df_filter[date_col].apply(lambda x: str(x)[:4]) == year, 'year_{}'.format(year)] = 1
        calendar_variables.append('year_{}'.format(year))

    print("Filtering dataframe to last date available...")
    var_cols = df_filter[df_filter[date_col] <= last_date].var().to_frame().reset_index()
    var_cols.columns = ['name', 'value']
    var_cols = var_cols[~(var_cols['name'].isin(default_variables + ['tag_forecast']))]
    var_cols = var_cols[~(var_cols['name'].isin([target]))]
    var_cols = var_cols[var_cols['value'] <= 1e-10]['name'].to_list()

    # print("Removing coluns without variation: ", var_cols)
    print("Removing coluns without variation... ")
    df_filter = df_filter.drop(var_cols, axis=1)

    calendar_variables = list(set(calendar_variables) - set(var_cols))

    features_variables = list(set(features_variables) - set(var_cols))

    df_filter['cons'] = 1
    features_variables.append("cons")

    print('Lags generates earlys rows with NAN')
    print('Filter early rows')
    df_filter = df_filter.iloc[6:, :]


    df_train = df_filter[df_filter[date_col] <= last_date]
    df_test = df_filter[df_filter['data'] >= date_pred]


    df_na_train = df_train.isna().sum().to_frame().rename(columns={0: 'n'}).reset_index()
    df_na_test = df_test.isna().sum().to_frame().rename(columns={0: 'n'}).reset_index()



    na_cols_train = [x for x in df_na_train[df_na_train['n'] > 0]['index'].to_list() if x != target]
    na_cols_test = [x for x in df_na_test[df_na_test['n'] > 0]['index'].to_list() if x != target]

    na_cols = na_cols_train + na_cols_test
    na_cols = list(dict.fromkeys(na_cols))

    # print('Removing coluns with all missing:', na_cols_train)
    df_train = df_train.drop(na_cols, axis=1)
    df_test = df_test.drop(na_cols, axis=1)

    df_test = df_test[df_train.columns]

    print("Pos processing format:", df_train.shape)

    return df_train, df_test

