import numpy as np
import pandas as pd
import cdsapi
import os

# Test file to 
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

pressure_level_variables = ['specific_rain_water_content'#, 
	#'specific_snow_water_content', 
	#'temperature',
	#'u_component_of_wind',
	#'v_component_of_wind',
	#'specific_humidity',
	#'vertical_velocity'
]


# land_variables #use for era5 land?

min_lat = np.inf
max_lat = -np.inf
min_lon = np.inf
max_lon = -np.inf

min_start = np.inf
max_end = -np.inf

# for each station
for i, station in stations.iterrows():
    if station['Downloaded']: # do for each usable station
        # find time and location bounaries for data download
    
        # get lat/lon
        lat = station.LATITUDE
        lon = station.LONGITUDE

        if lat < min_lat:
            min_lat = lat
        if lat > max_lat:
            max_lat = lat

        if lon < min_lon:
            min_lon = lon
        if lon > max_lon:
            max_lon = lon
    
        # get latest endtime and earliest starttime
        start_time = max(station['SD Start'], station['PC Start'])
        end_time = min(station['SD End'], station['PC End'])

        #if start_time < min_start:
        #    min_start = start_time
        #if end_time > max_end:
        #    max_end = end_time

# Note: not robut at dateline/poles

dataset = 'reanalysis-era5-pressure-levels'
product_type = 'reanalysis'

test = True # test with a smaller time range
if test:
    # smaller date range for test download (NOTE: 1/1000 of total time range)
    date = '2021-10-01/2022-5-31'
    filename = 'weather_test'

else:
    date = ''
    filename = 'weather'

time = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

format = 'netcdf'

pressure_level = ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200', '175', '150', '125', '100']
#pressure_level = ['1000', '925', '850', '700', '500', '400', '300', '250', '200']
#pressure_level = ['1000', '975', '950', '925']

#area = [str(max_lat + .25), str(min_lon - .25), str(min_lat  - .25), str(max_lon + .25)]
area = [51, -125, 48.2, -114.8]

print(area)



cds.retrieve(dataset, {
           "variable": pressure_level_variables,
           "pressure_level": pressure_level,
           "product_type": product_type,
           "date": date,
           "time": time,
           "format": format,
           "area": area
        }, filename+'.nc')

        




        # make api call with era5 dataset (possibly land?), centered at lat/lon, size big enough for 4 grid squares?, 
        # all/multiple pressure levels, correct time frame, all desired variables, as __ filetype, hourly?

        # save era5 data as csv.

        
# test
#cds.retrieve('reanalysis-era5-pressure-levels', {
#           "variable": "temperature",
#           "pressure_level": "1000",
#           "product_type": "reanalysis",
#           "date": "2017-12-01/2017-12-31",
#           "time": "12:00",
#           "format": "grib"
#        }, 'download.grib')
