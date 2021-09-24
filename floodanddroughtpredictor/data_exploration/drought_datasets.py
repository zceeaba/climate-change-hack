import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

from floodanddroughtpredictor.utils.path_utils import get_project_root


def extract_drough_data():
    # Here we have the drought index for all regions in Africa, we need to filter this to just those within Niger
    dnc = xr.open_dataset('/data/raw_spei/spei01.nc')
    df1 = dnc.to_dataframe()
    df1.reset_index(inplace = True)
    df1.to_csv('data/spei01.csv')

    # Read in GeoData for Niger
    niger_admin = gpd.read_file(
        r"data/ner_adm03_feb2018/NER_adm03_feb2018.shp"
    )
    niger_bound_coordinates = niger_admin['geometry']
    minx = niger_bound_coordinates.bounds.minx.min()
    miny = niger_bound_coordinates.bounds.miny.min()
    maxx = niger_bound_coordinates.bounds.maxx.max()
    maxy = niger_bound_coordinates.bounds.maxy.max()

    # Filter the massive dataset to only include lon's and lat's within range
    idx = np.where(
        (df1['lon'] >= minx) & (df1['lon'] <= maxx) & (df1['lat'] >= miny) & (df1['lat'] <= maxy)
    )

    df_niger = df1.loc[idx]

    # Remove the huge dataframe to reduce memory
    del df1

    df_niger.time = pd.to_datetime(df_niger["time"]).dt.strftime('%Y-%m-%d')
    df_niger.to_csv('data/spei01_niger.csv')

    geom = [Point(x, y) for x, y in zip(df_niger['lon'], df_niger['lat'])]
    niger_drought = gpd.GeoDataFrame(df_niger, geometry = geom)
    print(niger_drought.head())
    niger_drought.to_file(os.path.join(get_project_root(), "data/niger_drought.shp"))
