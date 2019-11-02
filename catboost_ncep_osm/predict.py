import gzip
import os
import random
import sys

import catboost
import geopandas
import numpy as np
import pandas
import pandas as pd
import xarray
from shapely.geometry import box

DATASETS_PATH = os.environ.get('DATASETS_PATH', '../data/')
MODELS_PATH = os.path.dirname(os.path.realpath(__file__))
OSM_GEO_DATA = os.path.join(MODELS_PATH, 'russia.osm.gpkg.gz')

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


def load_ncep_var(var, press_level):
    result = []
    year = 2019
    dataset_filename = '{}/ncep/{}.{}.nc'.format(DATASETS_PATH, var, year)
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

    df['lat'] = np.round(df.lat / 2.5).astype(np.int8)
    df['lon'] = np.round(df.lon / 2.5).astype(np.int8)
    return df.copy()


def add_ncep_features(df):
    df['lon'] = np.round(df.longitude / 2.5).astype(np.int8)
    df['lat'] = np.round(df.latitude / 2.5).astype(np.int8)
    for var, press_level in (('air', 1000), ('uwnd', 1000), ('rhum', 1000)):
        var_df = load_ncep_var(var, press_level)
        mdf = df.reset_index().merge(var_df, left_on=['lon', 'lat', 'date'], right_on=['lon', 'lat', 'time'],
                                     how='left', ).set_index('fire_id')
        for suffix in ('', '_7d', '_14d', '_30d'):
            df[var + suffix] = mdf[var + suffix]
    df.drop(['lon', 'lat'], axis=1, inplace=True)


def add_osm_features(df):
    with gzip.open(OSM_GEO_DATA, 'rb') as f:
        osm_df = geopandas.read_file(f, crs="epsg:4326")
    POINT_SIZE_X = 0.1
    POINT_SIZE_Y = 0.1
    geo_df = df.reset_index()
    geo_df = geopandas.GeoDataFrame(
        geo_df[['fire_id']],
        geometry=geo_df.apply(lambda x: box(
            x.longitude - POINT_SIZE_X / 2, x.latitude - POINT_SIZE_Y / 2,
            x.longitude + POINT_SIZE_X / 2, x.latitude + POINT_SIZE_Y / 2
        ), axis=1), crs="epsg:4326")

    geo_features = geopandas. \
        sjoin(geo_df, osm_df.drop(['ids', 'names'], axis=1), how='left', op='intersects'). \
        drop(['geometry', 'index_right'], axis=1). \
        groupby('fire_id'). \
        mean().fillna(0)

    for col in geo_features.columns:
        df[col] = geo_features[col]


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    reseed()

    df_test = pd.read_csv(input_csv, parse_dates=['date'])
    preprocess(df_test)
    add_ncep_features(df_test)
    add_osm_features(df_test)

    clf = catboost.CatBoostClassifier()
    clf.load_model(os.path.join(MODELS_PATH, 'catboost.cbm'))
    df_test.drop(['date'], axis=1, inplace=True)
    df_predictions = pandas.DataFrame(
        clf.predict_proba(df_test),
        index=df_test.index,
        columns=[
            'fire_{}_prob'.format(class_id)
            for class_id in range(1, 12)
        ],
    )

    df_predictions.to_csv(output_csv, index_label='fire_id')
