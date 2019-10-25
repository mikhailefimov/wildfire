import os
import random
import sys

import catboost
import numpy as np
import pandas
import pandas as pd

MODELS_PATH = os.path.dirname(os.path.realpath(__file__))

SEED = 42


def reseed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)


def preprocess(df):
    df['longitude'] = df['longitude'].astype(np.float32)
    df['latitude'] = df['latitude'].astype(np.float32)
    df['weekday'] = df.date.dt.weekday.astype(np.int8)
    df['month'] = df.date.dt.month.astype(np.int8)
    df.set_index('fire_id', inplace=True)
    df.drop(['fire_type', 'fire_type_name'], axis=1, inplace=True, errors='ignore')


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    reseed()

    df_test = pd.read_csv(input_csv, parse_dates=['date'])
    preprocess(df_test)

    clf = catboost.CatBoostClassifier()
    clf.load_model(os.path.join(MODELS_PATH, 'catboost.cbm'))
    df_test.drop(['date'], axis=1, inplace=True)
    df_predictions = pandas.DataFrame(
        clf.predict_proba(df_test),
        index=df_test.index,
        columns=[
            'fire_{}_prob'.format(class_id)
            for class_id in range(1, 12)
        ]
    )

    df_predictions.to_csv(output_csv, index_label='fire_id')
