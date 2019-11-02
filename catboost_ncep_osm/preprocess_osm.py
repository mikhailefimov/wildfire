import gzip
import os
import sys

import gdal
import geopandas
import numpy as np
import ogr
import pandas as pd
from shapely import wkb
from tqdm import tqdm

FIELDS = {'place', 'landuse', 'natural', 'building'}


def rebuild_osm_data(input_file, output_file):
    gdal.SetConfigOption('OGR_INTERLEAVED_READING', 'YES')
    osm = ogr.Open(input_file)
    geometry = []
    names = []
    ids = []
    fields = {}
    for i in range(osm.GetLayerCount()):
        L = osm.GetLayer(i)
        L.ResetReading()
        progress = tqdm(desc="Layer %i (%s)" % (i + 1, L.GetDescription()), mininterval=1)
        for feat in L:
            f_id = len(ids)
            for k, v in feat.items().items():
                if k not in FIELDS or v is None:
                    continue
                if f_id == len(ids):
                    geometry.append(wkb.loads(feat.geometry().ExportToWkb()))
                    names.append(feat.GetField('name') or "")
                    ids.append(feat.GetField('osm_id'))
                if v not in fields:
                    fields[v] = []
                fields[v].append(f_id)
            progress.update()
        progress.close()
    df = pd.DataFrame({'ids': ids, 'names': names, 'geometry': geometry})
    for field in fields:
        c = np.zeros_like(df.ids, dtype=np.int8)
        c[fields[field]] = 1
        df[field] = c
    osm_df = geopandas.GeoDataFrame(df, geometry='geometry', crs="epsg:4326")
    with gzip.open(output_file, 'wb') as f:
        osm_df.to_file(f, driver="GPKG")


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if os.path.isfile(output_file):
        print("File %s already exists, exiting." % output_file)
        exit(0)

    if not os.path.isfile(input_file):
        print("File %s not found." % input_file)
        print("You can download it from http://download.geofabrik.de/russia-latest.osm.pbf  (~2.5Gb)")
        exit(-1)

    print("=" * 40)
    print("This may take about one hour...")
    print("=" * 40)
    rebuild_osm_data(input_file, output_file)
