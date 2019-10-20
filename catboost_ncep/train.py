import os
import random

import catboost
import numpy as np
import pandas as pd
import xarray
from sklearn.metrics import roc_auc_score

SEED = 42
VAL_MONTHS = 6

ITERATIONS = 1000

PWD = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PWD, '../data')
MODELS_PATH = PWD


def reseed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)


def evaluate(y_true, y_pred):
    gt = np.zeros_like(y_pred, dtype=np.int8)
    gt[np.arange(y_true.shape[0]), y_true - 1] = 1
    result = {'roc_auc_micro': roc_auc_score(gt, y_pred, average='micro')}
    for ft in range(1, 12):
        gt = (y_true == ft)
        if gt.max() == gt.min():
            roc_auc = 0
        else:
            roc_auc = roc_auc_score(gt, y_pred[:, ft - 1])
        result[f'roc_auc_{ft}'] = roc_auc
    return result


def preprocess(df):
    df['longitude'] = df['longitude'].astype(np.float32)
    df['latitude'] = df['latitude'].astype(np.float32)
    df['weekday'] = df.date.dt.weekday.astype(np.int8)
    df['month'] = df.date.dt.month.astype(np.int8)
    df['ym'] = (df.date.dt.month + (df.date.dt.year - 2000) * 12).astype(np.int16)
    df['fire_type'] = df.fire_type.astype(np.uint8)
    df.set_index('point_id', inplace=True)
    df.drop(['fire_type_name'], axis=1, inplace=True)


def load_ncep_var(var, press_level):
    result = []
    for year in range(2012, 2020):
        dataset_filename = os.path.join(DATA_PATH, 'ncep', f'{var}.{year}.nc')
        ds = xarray.open_dataset(dataset_filename)
        ds = ds.sel(drop=True, level=press_level)[var]
        ds = ds[:, (ds.lat >= 15 * 2.5 - 0.1) & (ds.lat <= 29 * 2.5 + 0.1),
             (ds.lon >= 6 * 2.5 - 0.1) & (ds.lon <= 71 * 2.5 + 0.1)]
        result.append(ds)
    ds = xarray.merge(result)
    df = ds.to_dataframe()[[var]].reset_index()

    df = df.merge(ds.rolling(time=7).mean().to_dataframe()[[var]].reset_index(),
                  on=['lon', 'lat', 'time'], suffixes=('', '_7d'), how='left')
    df = df.merge(ds.rolling(time=14).mean().to_dataframe()[[var]].reset_index(),
                  on=['lon', 'lat', 'time'], suffixes=('', '_14d'), how='left')
    df = df.merge(ds.rolling(time=30).mean().to_dataframe()[[var]].reset_index(),
                  on=['lon', 'lat', 'time'], suffixes=('', '_30d'), how='left')

    df['lat'] = (df.lat / 2.5).astype(np.int8)
    df['lon'] = (df.lon / 2.5).astype(np.int8)
    return df.copy()


def add_ncep_features(df):
    df['lon'] = np.round(df.longitude / 2.5).astype(np.int8)
    df['lat'] = np.round(df.latitude / 2.5).astype(np.int8)
    for var, press_level in (('air', 1000), ('uwnd', 1000), ('rhum', 1000)):
        var_df = load_ncep_var(var, press_level)
        mdf = df.reset_index().merge(var_df, left_on=['lon', 'lat', 'date'], right_on=['lon', 'lat', 'time'],
                                     how='left', ).set_index('point_id')
        for suffix in ('', '_7d', '_14d', '_30d'):
            df[var + suffix] = mdf[var + suffix]
    df.drop(['lon', 'lat'], axis=1, inplace=True)


def prepare_dataset(filename):
    df = pd.read_csv(filename, parse_dates=['date'])
    preprocess(df)
    add_ncep_features(df)
    return df


def train_model(df_train):
    last_month = df_train.ym.max()
    train = df_train[df_train.ym <= last_month - VAL_MONTHS]
    val = df_train[df_train.ym > last_month - VAL_MONTHS]
    X_train = train.drop(['fire_type', 'ym', 'date'], axis=1)
    Y_train = train.fire_type
    X_val = val.drop(['fire_type', 'ym', 'date'], axis=1)
    Y_val = val.fire_type
    clf = catboost.CatBoostClassifier(loss_function='MultiClass',
                                      verbose=10, random_state=SEED, iterations=ITERATIONS)
    clf.fit(X_train, Y_train, eval_set=(X_val, Y_val))
    pred_train = clf.predict_proba(X_train)
    pred_val = clf.predict_proba(X_val)
    train_scores = evaluate(Y_train, pred_train)
    val_scores = evaluate(Y_val, pred_val)
    print("Train scores:")
    for k, v in train_scores.items():
        print("%s\t%f" % (k, v))
    print("Validation scores:")
    for k, v in val_scores.items():
        print("%s\t%f" % (k, v))
    clf.save_model(os.path.join(MODELS_PATH, 'catboost.cbm'))


if __name__ == '__main__':
    reseed()
    df_train = prepare_dataset(os.path.join(DATA_PATH, 'wildfires_train.csv'))
    train_model(df_train)
