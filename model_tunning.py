#import bibs for models

import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error



import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")

def models_tuning(x_train, y_train):
    print('choosing hyperparameters')
    def objective_ridge(trial):

        tss_cv = TimeSeriesSplit(n_splits=5)
        # Step 2. Setup values for the hyperparameters:
        alpha = trial.suggest_float("alpha", low=1e-20, high=10000, log=True) #Log ou Step, somente um dos dois
        intercept = trial.suggest_categorical("fit_intercept", [True, False])
        tol = trial.suggest_float("tol", 0.001, 0.01, log=True)
        solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "saga", "sag"])

        ## Create Model
        regressor = Ridge(alpha=alpha, fit_intercept=intercept, tol=tol, solver=solver)

        # Step 3: Scoring method:
        score = cross_val_score(regressor, x_train, y_train, n_jobs=-1, cv=tss_cv, scoring='r2')
        accuracy = score.mean()
        return accuracy


    def objective_lgbm(trial):
        tss_cv = TimeSeriesSplit(n_splits=5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
        num_leaves = trial.suggest_int('num_leaves', 2, 256)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 100)
        bagging_fraction  = trial.suggest_uniform('bagging_fraction', 0.1, 1.0)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
        tss_cv = TimeSeriesSplit(n_splits=5)
        model_lgbm= LGBMRegressor(random_state=0, n_estimators=500, bagging_freq=1,
                                learning_rate=learning_rate, num_leaves=num_leaves,
                                min_data_in_leaf=min_data_in_leaf, bagging_fraction=bagging_fraction,
                                colsample_bytree=colsample_bytree, verbose = 0)

        score = cross_val_score(model_lgbm, x_train, y_train, n_jobs=-1, cv=tss_cv, scoring='neg_mean_squared_error')
        accuracy = score.mean()
        return accuracy


    def objective_rf(trial):
        tss_cv = TimeSeriesSplit(n_splits=5)
        rf_n_estimators = trial.suggest_int("n_estimators", 100, 5000, step=100)
        rf_max_depth = trial.suggest_int("max_depth", 2, 6)

        max_features = trial.suggest_categorical( "max_features", ["sqrt", "log2"])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 4, step = 1)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step = 2)


        rf = RandomForestRegressor(n_estimators = rf_n_estimators,
                                   max_depth = rf_max_depth, max_features=max_features,
                                   min_samples_leaf = min_samples_leaf,
                                   min_samples_split = min_samples_split,
                                   verbose = 0)
        score = cross_val_score(rf, x_train, y_train.values.ravel(), n_jobs=-1, cv=tss_cv, scoring='r2')
        accuracy = score.mean()
        return accuracy



    study_ridge = optuna.create_study(direction="minimize")
    study_ridge.optimize(objective_ridge, n_trials=100)

    study_lgbm = optuna.create_study(direction="minimize")
    study_lgbm.optimize(objective_lgbm, n_trials=100)

    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective_rf, n_trials=100)


    print('Creating Dict with best params and scores')

    dictionary_hyperparameters = {'Ridge': {'best_value': study_ridge.best_value, 'hyperparams': study_ridge.best_params},
                                  'LGBM': {'best_value': study_lgbm.best_value, 'hyperparams': study_lgbm.best_params},
                                  'RF': {'best_value': study_rf.best_value, 'hyperparams': study_rf.best_params}, }

    studies_trained = {'Ridge': study_ridge,
                        'LGBM':study_lgbm,
                        'RF':study_rf}



    return dictionary_hyperparameters, studies_trained


def training_full_data(dict_hyperparams, x_train, y_train, x_test, y_test):

    print('training Rigde model')
    Ridge_trained = Ridge().set_params(**dict_hyperparams['Ridge']['hyperparams'])
    Ridge_trained.fit(x_train, y_train)
    predict_ridge = Ridge_trained.predict(x_test)

    print('scores from Ridge')
    print("R2 score:", r2_score(y_test, predict_ridge))
    print("Mean Abs Percent Error:", mean_absolute_percentage_error(y_test, predict_ridge))

    print('training LGBM model')
    LGBM_trained = LGBMRegressor().set_params(**dict_hyperparams['LGBM']['hyperparams'])
    LGBM_trained.fit(x_train, y_train)
    predict_lgbm = LGBM_trained.predict(x_test)

    print('scores from LGBM')
    print("R2 score:", r2_score(y_test, predict_lgbm))
    print("Mean Abs Percent Error:", mean_absolute_percentage_error(y_test, predict_lgbm))

    print('training RF model')
    RF_trained = RandomForestRegressor().set_params(**dict_hyperparams['RF']['hyperparams'])
    RF_trained.fit(x_train, y_train)
    predict_rf = RF_trained.predict(x_test)

    print('scores from RF')
    print("R2 score:", r2_score(y_test, predict_rf))
    print("Mean Abs Percent Error:", mean_absolute_percentage_error(y_test, predict_rf))

    df_predicted = pd.concat([y_test.reset_index(), pd.DataFrame(predict_ridge, columns=["Ridge"]),
                               pd.DataFrame(predict_lgbm, columns=["LGBM"]),
                               pd.DataFrame(predict_rf, columns=["RF"])], axis=1)

    scores_dictionary = {"Ridge":{"R2":r2_score(y_test, predict_ridge), "MAPE":mean_absolute_percentage_error(y_test, predict_ridge)},
                         "LGBM":{"R2":r2_score(y_test, predict_lgbm), "MAPE":mean_absolute_percentage_error(y_test, predict_lgbm)},
                         "RF":{"R2":r2_score(y_test, predict_rf), "MAPE":mean_absolute_percentage_error(y_test, predict_rf)}}

    return df_predicted, scores_dictionary