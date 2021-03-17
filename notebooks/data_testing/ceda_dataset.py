import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Here we have the drought index for all regions in Africa, we need to filter this to just those within Niger
dnc = xr.open_dataset('data/spei01.nc')
df1 = dnc.to_dataframe()
df1.reset_index(inplace=True)
df1.head()




geom = [Point(x,y) for x, y in zip(df1['lon'], df1['lat'])]
gdf_1month = gpd.GeoDataFrame(df1, geometry=geom)
print(gdf_1month.head())


# Here we have the drought index for all regions in Africa, we need to filter this to just those within Niger





# dnc = xr.open_dataset('../downloads/spei48.nc')
# df48 = dnc.to_dataframe()
# df48.reset_index(inplace=True)
# df48
# df48.spei
#
# geom = [Point(x,y) for x, y in zip(df48['lon'], df48['lat'])]
# gdf_48month = gpd.GeoDataFrame(df48, geometry=geom)
# print(gdf_48month.head())




from netCDF4 import Dataset


def ncdump(nc_fid, verb=True):
    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    return nc_attrs, nc_dims, nc_vars


nc_f = '../downloads/spei48.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)



from matplotlib import pyplot as plt
import pandas as pd
import netCDF4
from netCDF4 import Dataset
import h5py

url ="https://dap.ceda.ac.uk/neodc/spei_africa/data/spei01.nc"
vname = 'SPEI'

nc = netCDF4.Dataset(url)
h = nc.variables[]
times = nc.variables['time']
jd = netCDF4.num2date(times[:],times.units)
hs = pd.Series(h[:,station],index=jd)

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111)
hs.plot(ax=ax,title='%s at %s' % (h.long_name,nc.id))
ax.set_ylabel(h.units)

h = nc.variables[vname]
times = nc.variables['time']
jd = netCDF4.num2date(times[:],times.units)
hs = pd.Series(h[:,station],index=jd)



