# import libraries
import geopandas as gpd
import pandas as pd
import re

from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt

# Read in GeoData for Niger and keep only relevant rows
niger_admin = gpd.read_file(r"data/ner_adm03_feb2018/NER_adm03_feb2018.shp")
niger_roads = gpd.read_file(r"data/nmb_roads_0/NE_Roads.shp")
niger_electricity = gpd.read_file(r"data/ner_powerplants/NER_PowerPlants.shp")
niger_power = gpd.read_file(r"data/niger-electricity-transmission-network/Niger Electricity Transmission Network.shp")

africopolis = gpd.read_file(r"data/Africapolis_2015_shp/africapolis.shp")
africopolis = africopolis[africopolis.ISO=='NER']

niger_drought = gpd.read_file(r"data/niger_drought_preds/niger_drought.shp")


# niger_admin[:10].plot()
# niger_roads[:10].plot()
# africopolis[:10].plot()

niger_roads.columns
niger_admin.columns
niger_admin.adm_02
africopolis.columns
africopolis.Name


niger_roads.crs
# <Geographic 2D CRS: EPSG:4326>
niger_admin.crs
# <Geographic 2D CRS: EPSG:4326>
niger_electricity.crs
# <Geographic 2D CRS: EPSG:4326>
niger_power.crs
# <Geographic 2D CRS: EPSG:4326>
africopolis.crs
# <Geographic 2D CRS: EPSG:4326>

fig, ax = plt.subplots(figsize = (20, 20))

# first setup the plot using the crop_extent layer as the base layer
niger_admin.plot(color='lightgrey',
                      edgecolor = 'black',
                      ax = ax,
                      alpha=.5)
# then add another layer using geopandas syntax .plot, and calling the ax variable as the axis argument
niger_roads.plot(ax=ax, cmap='Set2')
niger_electricity.plot(ax=ax, cmap='Set1')
niger_power.plot(ax=ax, cmap='Set3')
africopolis.plot(ax=ax, cmap='PuBuGn_r')
# add a title to the plot
ax.set_title('Roads and Admin Regions\nin Niger')
ax.set_axis_off()
plt.axis('equal')
plt.show()
