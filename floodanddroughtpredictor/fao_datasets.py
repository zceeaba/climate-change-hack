# import libraries
import geopandas as gpd
import pandas as pd
import re

from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt

employment = pd.read_csv(r'data/FAOSTAT/Employment_Indicators_E_All_Data_(Normalized)/Employment_Indicators_E_All_Data_(Normalized).csv')
temperature_change = pd.read_csv(r'data/FAOSTAT/Environment_Temperature_change_E_All_Data_(Normalized)/Environment_Temperature_change_E_All_Data_(Normalized).csv')
population = pd.read_csv(r'data/FAOSTAT/Population_E_All_Data_(Normalized)/Population_E_All_Data_(Normalized).csv')
crops = pd.read_csv(r'data/FAOSTAT/Production_CropsProcessed_E_All_Data_(Normalized)/Production_CropsProcessed_E_All_Data_(Normalized).csv')
land_cover = pd.read_csv(r'data/FAOSTAT/Environment_LandCover_E_All_Data_(Normalized)/Environment_LandCover_E_All_Data_(Normalized).csv')
land_use = pd.read_csv(r'data/FAOSTAT/Environment_LandUse_E_All_Data_(Normalized)/Environment_LandUse_E_All_Data_(Normalized).csv')
food_aid = pd.read_csv(r'data/FAOSTAT/Food_Aid_Shipments_WFP_E_All_Data_(Normalized)/Food_Aid_Shipments_WFP_E_All_Data_(Normalized).csv')
food_security = pd.read_csv(r'data/FAOSTAT/Food_Security_Data_E_All_Data_(Normalized)/Food_Security_Data_E_All_Data_(Normalized).csv')

employment.head()
employment.columns
employment = employment[employment.Area=='Niger']
employment['Indicator_Units'] = employment.Indicator + '_' + employment.Unit
employment['data_source'] = 'employment'
employment['time_granularity'] = 'years'
employment['geographic_level'] = 'country'
employment_wide = employment.pivot(index=('Area Code', 'Area', 'Source Code', 'Year'), columns='Indicator_Units')['Value']
employment_wide.reset_index(inplace=True)

temperature_change.head()
temperature_change.columns
temperature_change = temperature_change[temperature_change.Area=='Niger']
temperature_change['Indicator_Units'] = temperature_change.Element + '_' + temperature_change.Unit
temperature_change['data_source'] = 'temperature'
temperature_change['time_granularity'] = 'months'
temperature_change['geographic_level'] = 'country'
temperature_change_wide = temperature_change.pivot(index=('Area Code', 'Area', 'Element Code', 'Year', 'Months'), columns='Indicator_Units')['Value']
temperature_change_wide.reset_index(inplace=True)


population.head()
population.columns
population = population[population.Area=='Niger']
population['Indicator_Units'] = population.Element + '_' + population.Unit
population['data_source'] = 'population'
population['time_granularity'] = 'years'
population['geographic_level'] = 'country'
population_wide = population.pivot(index=('Area Code', 'Area', 'Element Code', 'Year'), columns='Indicator_Units')['Value']
population_wide.reset_index(inplace=True)

crops.head()
crops.columns
crops = crops[crops.Area=='Niger']
crops['Indicator_Units'] = crops.Element + '_' + crops.Unit
crops['data_source'] = 'crops'
crops['time_granularity'] = 'years'
crops['geographic_level'] = 'country'
crops_wide = crops.pivot(index=('Area Code', 'Area', 'Element Code', 'Item', 'Year'), columns='Indicator_Units')['Value']
crops_wide.reset_index(inplace=True)

land_cover.head()
land_cover.columns
land_cover = land_cover[land_cover.Area=='Niger']
land_cover['Indicator_Units'] = land_cover.Element + '_' + land_cover.Unit
land_cover['data_source'] = 'land_cover'
land_cover['time_granularity'] = 'years'
land_cover['geographic_level'] = 'country'
land_cover_wide = land_cover.pivot(index=('Area Code', 'Area', 'Element Code', 'Item', 'Year'), columns='Indicator_Units')['Value']
land_cover_wide.reset_index(inplace=True)

land_use.head()
land_use.columns
land_use = land_use[land_use.Area=='Niger']
land_use['Indicator_Units'] = land_use.Element + '_' + land_use.Unit
land_use['data_source'] = 'land_use'
land_use['time_granularity'] = 'years'
land_use['geographic_level'] = 'country'
land_use_wide = land_cover.pivot(index=('Area Code', 'Area', 'Element Code', 'Item', 'Year'), columns='Indicator_Units')['Value']
land_use_wide.reset_index(inplace=True)

food_aid.head()
food_aid.columns
food_aid = food_aid[food_aid.Area=='Niger']
food_aid['Indicator_Units'] = food_aid.Element + '_' + food_aid.Unit
food_aid['data_source'] = 'food_aid'
food_aid['time_granularity'] = 'years'
food_aid['geographic_level'] = 'country'
food_aid_wide = food_aid.pivot(index=('Area Code', 'Area', 'Element Code', 'Item', 'Year'), columns='Indicator_Units')['Value']
food_aid_wide.reset_index(inplace=True)

food_security.head()
food_security.columns
food_security = food_security[food_security.Area=='Niger']
food_security['Indicator_Units'] = food_security.Element + '_' + food_security.Unit
food_security['data_source'] = 'food_security'
food_security['time_granularity'] = 'years'
food_security['geographic_level'] = 'country'
food_security_wide = food_security.pivot(index=('Area Code', 'Area', 'Element Code', 'Item', 'Year'), columns='Indicator_Units')['Value']
food_security_wide.reset_index(inplace=True)

niger_data_long = pd.concat([employment, temperature_change, population, crops ], ignore_index=True)
niger_data_wide = pd.merge(employment_wide, temperature_change_wide, how='outer', on=('Area Code', 'Area', 'Year'))
niger_data_wide = pd.merge(niger_data_wide, population_wide, how='outer', on=('Area Code', 'Area', 'Year', 'Element Code'))
niger_data_wide = pd.merge(niger_data_wide, temperature_change_wide, how='outer', on=('Area Code', 'Area', 'Year', 'Element Code'))

niger_data_long.to_csv('data/processed_faodata_long.csv')
niger_data_wide.to_csv('data/processed_faodata_wide.csv')