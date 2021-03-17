import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import basemap
from faodata import faodownload

database_id = 'faostat'
datasets = faodownload.get_datasets(database_id)
# Download data
database_id = 'faostat'
dataset_id = 'live-prod'
field_id = 'm5111'
year = 2013
data = faodownload.get_data(database_id, dataset_id, field_id, year=year)

# Select data
item = 'Cattle'
idx = data['Item'] == item
data = data.loc[idx, ['country', 'value']]

# Instantiate matplotlib and basemap objects
plt.close('all')
fig, ax = plt.subplots()
map = basemap.Basemap(projection='robin', \
        lon_0=10, lat_0=50, ax = ax)

map.drawcoastlines(color='grey')
map.drawcountries(color='grey')

# Categorize data according to percentiles
cat = [np.percentile(data['value'], pp) \
        for pp in range(10, 100, 10)]

# Draw plot
faomap.plot(map, data, cat, ndigits=0)

map.ax.legend(loc=3)
ax.set_title('%s population, %d' % (item, year),
        fontsize=15)

# Add a footer to the figure to
# indicate data source
faomap.mapfooter(fig, database_id, dataset_id, field_id)