import xarray as xr

dataset = xr.open_dataset('data/weather/weather_upper/weather_upper_2013_region_3_specific_rain_water_content.nc')

print(dataset)