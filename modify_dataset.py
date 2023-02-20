# modifies dataset to it's the right size/variables to train on
import numpy as np
import pandas as pd
import os
import xarray as xr
import datetime as dt
import scipy

filename = "data/datasets/" + "test_data_4.csv"
outfilename = "data/datasets/" + "test_data_4_modified.csv"

data = pd.read_csv(filename)
print(data)

pressure_level_variables = ['specific_rain_water_content', 
	'specific_snow_water_content', 
	'temperature',
	'u_component_of_wind',
	'v_component_of_wind',
	'specific_humidity',
	'vertical_velocity'
]

surface_variables = ['angle_of_sub_gridscale_orography', 
    'slope_of_sub_gridscale_orography', 
    'instantaneous_10_metre_wind gust',
    '10m_u_component_of_wind',
    '10m_v_component_of_neutral_wind',
    'surface_pressure',
	'2m_temperature',
    '2m_dewpoint_temperature',
	'100m_u_component_of_wind',
	'100m_v_component_of_wind',
	'precipitation_type',
	'surface_solar_radiation_downwards',
	'forecast_albedo',
	'snow_evaporation',
	'mean_snow_evaporation_rate',
	'mean_snowmelt_rate',
	'mean_surface_downward_short_wave_radiation_flux',
	'mean_surface_downward_long_wave_radiation_flux',
	'mean_top_net_short_wave_radiation_flux',
	'mean_top_net_long_wave_radiation_flux',
    'snowfall',
    'total_precipitation'
]

pressure_levels = ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200']

pressure_variables_to_drop = ['specific_rain_water_content', 
	'specific_snow_water_content']
surface_variables_to_drop = ['angle_of_sub_gridscale_orography', 
    'slope_of_sub_gridscale_orography', 
    'surface_pressure',
    '100m_u_component_of_wind',
	'100m_v_component_of_wind',
	'precipitation_type',
	'surface_solar_radiation_downwards',
	'forecast_albedo',
	'snow_evaporation',
	'mean_surface_downward_short_wave_radiation_flux',
	'mean_surface_downward_long_wave_radiation_flux',
	'mean_top_net_short_wave_radiation_flux',
	'mean_top_net_long_wave_radiation_flux'
    ]

pressure_levels_to_drop = ['975', '950', '875', '825', '800', '775', '750', '650', '600', '550', '450', '350', '225']

# drop unwanted columns to decrease dataset size
#data = data.drop(columns = surface_variables_to_drop)
pressure_columns_to_drop = []

for pl in pressure_levels_to_drop:
    for var in pressure_level_variables:
        pressure_columns_to_drop = np.append(pressure_columns_to_drop, [var + '_' + pl])

for pl in pressure_levels:
    for var in pressure_variables_to_drop:
        pressure_columns_to_drop = np.append(pressure_columns_to_drop, [var + '_' + pl])

pressure_columns_to_drop = list(dict.fromkeys(pressure_columns_to_drop))


#data = data.drop(columns = pressure_columns_to_drop)




data = data[data['elevation'].notna()] #remove stations with no elevation data
data = data[data['snowfall'] != 0] #remove examples with no reanalysis snowfall
#remove examples with PC or SD values below desired
data = data[data['pc_increase'] > 3]
data = data[data['sd_increase'] > 3]

data = data.drop(columns = ['starttime', 'endtime', 'Unnamed: 0', 'Unnamed: 0.1', 'pc_increase', 'sd_increase'])
print(data)




data.to_csv(outfilename)
