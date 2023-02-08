# pull in stations data

import numpy as np
import pandas as pd
import os
import xarray as xr
import datetime as dt
import scipy

# This file plots station points and rectangles to help decide where to get data from
# ORDER: NONE

# download stations data
data = pd.read_csv("data/stations_new.csv")   
stations = pd.DataFrame(data)

pressure_level_variables = ['specific_rain_water_content', 
	'specific_snow_water_content', 
	'temperature',
	'u_component_of_wind',
	'v_component_of_wind',
	'specific_humidity',
	'vertical_velocity'
]

pressure_level_variables_shortnames = ['crwc', 
	'cswc', 
	't',
	'u',
	'v',
	'q',
	'w'
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

surface_variables_shortnames = ['anor', 'slor', 'i10fg', 'u10', 'v10n', 'sp', 't2m', 'd2m', 'u100', 'v100', 'ptype', 'ssrd', 'fal', 'es', 'mser', 'msmr', 'msdwswrf', 'msdwlwrf', 'mtnswrf', 'mtnlwrf', 'sf', 'tp']

# TODO: consider how high up to go
pressure_levels = ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200']

pc_filestart = "DataSetExport-PC.Working@"
sd_filestart = "DataSetExport-SD.Working@"

# helper functions:
# converts the time as as string to a datetime variable
def convertToDatetime(time_str, start):
    year = int(time_str.split("/", 2)[2][0:4])
    month = int(time_str.split("/", 1)[0])
    day = int(time_str.split("/", 1)[1].split("/", 1)[0])
    return dt.datetime(year, month, day)

# checks if recorded precip and snow increase within the given window. If so and SLR is reasonable, return the SLR
def checkValidWindow(window_start, window_end, pc_data, sd_data):

    # invalid if in a month that we don't have weather data for
    if (window_start.month > 5 and window_start.month < 10):
        return [False, 0]
    if (window_start.month > 5 and window_start.year >= 2022):
        return [False, 0]
    if (window_start.month == 5 and window_start.day == 31 and window_start.hour == 18):
        return [False, 0]
    try:

        pc_start = pc_data.loc[window_start]["Value (mm)"]
        pc_end = pc_data.loc[window_end]["Value (mm)"]
        sd_start = sd_data.loc[window_start]["Value (cm)"]
        sd_end = sd_data.loc[window_end]["Value (cm)"]
        pc_increase = pc_end - pc_start
        sd_increase = sd_start - sd_end
        if (pc_increase == 0 or sd_increase == 0):
            return [False, 0]

        SLR = sd_increase/pc_increase * 10

        # TODO check raw data for outliers
        if (pc_increase < 1 or sd_increase < 1 or SLR < 2 or SLR > 50 or sd_increase > 50 or pc_increase > 50):
            return [False, 0]
        #print("precip (mm): " + str(pc_increase))
        #print("snow(cm): " + str(sd_increase))
        return [True, SLR]
    except: # no data exists in the given window, so there is no valid data to return
        return [False, 0]

# returns the year number of the file containing data at start_time
def getFileYear(start_time):
    if start_time.month >=10:
        year = start_time.year
    else:
        year = start_time.year - 1
    return year

# returns dataset that includes pressure level weather data at given lat, lon, year
def openWeatherUpper(lat, lon, start_time):
    if (lat < 51):
        region = 1
    elif (lon > -123):
        region = 2
    else:
        region = 3
    year = getFileYear(start_time)
    first = True
    for variable in pressure_level_variables:
        filepath = 'data/weather/weather_upper/weather_upper_' + str(year) + '_region_' + str(region) + '_' + variable + '.nc'
        if (first):
            # print('zero')
            first = False
            dataset = xr.open_dataset(filepath)
           # print(dataset)
        else:
            current_dataset = xr.open_dataset(filepath)
            # print(current_dataset)
            dataset = xr.merge([dataset, current_dataset])
    #print(dataset)
    return dataset

# returns dataset that includes land weather data at given lat, lon, time
def openWeatherLand(lat, lon, start_time):
    if (lat < 51):
        region = 1
    elif (lon > -123):
        region = 2
    else:
        region = 3
    year = getFileYear(start_time)
    filepath = 'data/weather/weather_land/weather_land_' + str(year) + '_region_' + str(region) + '.nc'
    dataset = xr.open_dataset(filepath)
    #print("KEYS!")
    #print(list(dataset.keys()))
    return(dataset)
    

pc_filestart = "DataSetExport-PC.Working@"
sd_filestart = "DataSetExport-SD.Working@"

# keeps track of total number of valid time windows
total_windows = 0

delta = dt.timedelta(hours=6)

columns = []
# create empty dataframe that we will put all the data into:
for variable in pressure_level_variables:
    for pl in pressure_levels:
        columns = columns + [variable + '_' + pl]
columns = columns + surface_variables + ['elevation', 'latitude', 'longitude', 'SLR']
#print(columns)
df = pd.DataFrame(columns=columns)
row_num = 0
#print(df)


# for each station
for i, station in stations.iterrows():
    if station['Downloaded']: # do for each usable station
        print("Station " + str(i) + " of 122")

        # get lat/lon
        lat = station.LATITUDE
        lon = station.LONGITUDE
        elevation = station.ELEVATION

        # get latest endtime and earliest starttime
        start_time = max(station['SD Start'], station['PC Start'])
        end_time = min(station['SD End'], station['PC End'])
        start_time = convertToDatetime(start_time, True)
        end_time = convertToDatetime(end_time, False)

        window_start = start_time
        window_end = start_time + delta

        # open SD and PC data
        pc_filename = pc_filestart + station.LCTN_ID
        for file in os.listdir("data/snow/"):
            if file.startswith(pc_filename): # file is name of each pc data file
                pc_full_filename = file
        
        data = pd.read_csv("data/snow/" + pc_full_filename, skiprows=2, parse_dates=["Timestamp (UTC)"]) 
        pc_data = pd.DataFrame(data)

        sd_filename = sd_filestart + station.LCTN_ID
        for file in os.listdir("data/snow/"):
            if file.startswith(sd_filename): # file is name of each sd data file
                sd_full_filename = file
        
        data = pd.read_csv("data/snow/" + sd_full_filename, skiprows=2, parse_dates=["Timestamp (UTC)"]) 
        sd_data = pd.DataFrame(data)

        # make titles readable and round all recorded times to the nearest hour 
        pc_data["Timestamp (UTC)"] = pd.to_datetime(pc_data["Timestamp (UTC)"])
        sd_data["Timestamp (UTC)"] = pd.to_datetime(sd_data["Timestamp (UTC)"])

        pc_data["Timestamp (UTC)"] = pc_data["Timestamp (UTC)"].dt.round("H")
        sd_data["Timestamp (UTC)"] = sd_data["Timestamp (UTC)"].dt.round("H")

        pc_data = pc_data.set_index('Timestamp (UTC)')
        sd_data = sd_data.set_index('Timestamp (UTC)')

        fileyear = getFileYear(start_time)

        # open weather files
        weather_upper = openWeatherUpper(lat, lon, start_time)
        weather_land = openWeatherLand(lat, lon, start_time)

        valid_windows = 0
        
        # loop through all possible time windows
        while (window_end <= end_time):            
            
            [valid, SLR] = checkValidWindow(window_start, window_end, pc_data, sd_data)
            if valid: # there is an SLR recorded from this time window
                
                #print("Valid window! Starting at " + str(window_start))
                #print("SLR: " + str(SLR))
                valid_windows += 1

                # if we need to move to the next weather dataset, do it
                current_fileyear = getFileYear(window_start)
                if not(current_fileyear == fileyear):
                    fileyear = current_fileyear
                    weather_upper = openWeatherUpper(lat, lon, window_start)
                    weather_land = openWeatherLand(lat, lon, window_start)
                    print("updated files to be fileyear " + str(fileyear))
                
                # Grab weather data at that location and time and put into full data array, along with SLR

                df.at[row_num, 'SLR'] = SLR
                df.at[row_num, 'elevation'] = elevation
                df.at[row_num, 'latitude'] = lat
                df.at[row_num, 'longitude'] = lon
                
                # 
                # pressure level 

                # this way is slower
                #for pl in pressure_levels:
                #    for (variable, var_name) in zip(pressure_level_variables_shortnames, pressure_level_variables):
                #        # get value (at start, middle, and end of time window)
                #        val1 = weather_upper.sel(time = window_start).sel(level = int(pl)).interp(latitude = lat, longitude = lon)[variable].values
                #        val2 = weather_upper.sel(time = window_start+delta/2).sel(level = int(pl)).interp(latitude = lat, longitude = lon)[variable].values
                #        val3 = weather_upper.sel(time = window_end).sel(level = int(pl)).interp(latitude = lat, longitude = lon)[variable].values
                #        # interpolate between times
                #        val = (val1 + 2*val2 + val3)/4
                #        df.at[row_num, var_name + '_' + pl] = val
                
                # TODO: check how it's interpolating-- look at what the options are there-- do bicubic?
                v1 = weather_upper.sel(time = window_start).interp(latitude = lat, longitude = lon, method='slinear')
                v2 = weather_upper.sel(time = window_start + delta/2).interp(latitude = lat, longitude = lon, method='slinear')
                #v3 = weather_upper.sel(time = window_end).interp(latitude = lat, longitude = lon, method='slinear')
                #v = (v1 + 2*v2 + v3)/4
                v = (v1 + v2)/2
                v = v.to_array().values.ravel()
                df.loc[row_num, 'specific_rain_water_content_1000':'vertical_velocity_200'] = v
                  

                # ground 
                #for (variable, var_name) in zip(surface_variables_shortnames, surface_variables):
                #    # get value (at start, middle, and end of time window)
                #    val1 = weather_land.sel(time = window_start).interp(latitude = lat, longitude = lon)[variable].values
                #    val2 = weather_land.sel(time = window_start+delta/2).interp(latitude = lat, longitude = lon)[variable].values
                #    val3 = weather_land.sel(time = window_end).interp(latitude = lat, longitude = lon)[variable].values
                #    # interpolate between times
                #    val = (val1 + 2*val2 + val3)/4
                #    df.at[row_num, var_name] = val

                v1 = weather_land.sel(time = window_start).interp(latitude = lat, longitude = lon, method='slinear')
                v2 = weather_land.sel(time = window_start+delta/2).interp(latitude = lat, longitude = lon, method='slinear')
                #v3 = weather_land.sel(time = window_end).interp(latitude = lat, longitude = lon, method='slinear')
                #v = (v1 + 2*v2 + v3)/4
                v = (v1 + v2)/2
                v = v.to_array().values.ravel()
                df.loc[row_num, 'angle_of_sub_gridscale_orography':'total_precipitation'] = v
                
                row_num += 1
                #print('row ' + str(row_num) + ' added')
                if (row_num%100 == 0):
                    df.to_csv('data/test_data.csv')  
                    ds = xr.Dataset.from_dataframe(df)
                    ds.to_netcdf('data/test_data.nc')  
                    print(str(row_num) + " rows saved, saved as files")

            
            # move to next time window:   
            window_start += delta
            window_end += delta 

        print(str(valid_windows) + " windows found at station")
        total_windows += valid_windows

# save entire dataset
print("Total examples:")
print(total_windows)
df.to_csv('data/test_data.csv')  
ds = xr.Dataset.from_dataframe(df)
ds.to_netcdf('data/test_data.nc')  
# save dataset

    #   for each six hour window
    #       if window has increasing precip and snow (by at least a certain amount)
    #           calculate weather 1d array for that window/location (pulling from appropriate files)
    #           if weather is not too warm or changing to fast
    #               add weather array to X and SLR to y
# 

# save overall dataset