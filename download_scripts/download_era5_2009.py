import numpy as np
import pandas as pd
import cdsapi
import os

# This file downloads ERA5 data in a region encompassing all stations for appropriate times
# ORDER: 2

# download stations data
data = pd.read_csv("data/stations_new.csv")   
stations = pd.DataFrame(data)

pc_filestart = "DataSetExport-PC.Working@"
sd_filestart = "DataSetExport-SD.Working@"

# setup cds api
cds = cdsapi.Client()

# list of variables to download
surface_variables = ['angle_of_sub_gridscale_orography', 
    'slope_of_sub_gridscale_orography', 
    'instantaneous_10 metre wind gust',
    '10m_u-component_of_wind',
    '10m_v-component_of_neutral_wind',
    'surface_pressure',
	'2m_temperature',
	'100m_u-component_of_wind',
	'100m_v-component_of_wind',
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
    #Some type of snowfall/precip rate (convective/large scale):
    'total_precip'
    # snow accumulation?
]

pressure_level_variables = ['specific_rain_water_content', 
	'specific_snow_water_content', 
	'temperature',
	'u_component_of_wind',
	'v_component_of_wind',
	'specific_humidity',
	'vertical_velocity'
]

dataset = 'reanalysis-era5-pressure-levels'
product_type = 'reanalysis'

time = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

format = 'netcdf'

pressure_level = ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200']

area1 = [51, -125, 48.25, -115]
area2 = [55.75, -123, 50.75, -118]
area3 = [58, -130.5, 50.75, -123.75]

regions = [area1, area2, area3]

year = 2009

date = str(year) + '-10-01/' + str(year+1) + '-5-31'
region = 0
for area in regions:
    region = region + 1

    # pressure level data
    for variable in pressure_level_variables:
        
        filename = "weather_upper_" + str(year) + "_region_" + str(region) + "_" + variable
        print("requesting" + filename)
        cds.retrieve(dataset, {
            "variable": [variable],
            "pressure_level": pressure_level,
            "product_type": product_type,
            "date": date,
            "time": time,
            "format": format,
            "area": area
        }, filename+'.nc')

        # land data
        #filename = "weather_land_" + str(year) + "_region_" + str(region)
        #print("requesting" + filename)
        #cds.retrieve('reanalysis-era5-single-levels', {
        #        "variable": surface_variables,
        #        "product_type": product_type,
        #        "date": date,
        #        "time": time,
        #        "format": format,
        #        "area": area
        #    }, filename+'.nc')
