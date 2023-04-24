import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cdsapi
import os
import matplotlib.patches as patches
import plotly.express as px

import numpy as np
import pandas as pd


stations = pd.read_csv('data/stations_new.csv')
print(stations)
stations = stations[stations.Downloaded == True]
stations_list = stations.drop(columns = ['SNWASWS_ID', 'STATUS', 'Unnamed: 0'])
stations_list = stations_list.iloc[:, :5]


print(stations_list.LATITUDE)

clusters = np.array([])
for i in range(len(stations_list.LATITUDE)):
    lat = stations_list.LATITUDE.iloc[i]
    lon = stations_list.LONGITUDE.iloc[i]
    if (lon < -122.5 and lat < 54):
        clusters = np.append(clusters, ['Coast Range'])
    elif (lon < -122.5 ):
        clusters = np.append(clusters, ['Rockies'])
    else:
        clusters = np.append(clusters, ['Northern BC'])

stations_list['Region'] = clusters

print(stations_list)

fig = px.scatter_geo(lat =stations_list.LATITUDE, lon = stations_list.LONGITUDE, color = stations_list.Region, title = "Station Locations Colored by Region and ERA5 Data Domains")
fig.add_trace(px.line_geo(lat = [51, 48.2, 48.2, 51, 51], lon = [-125, -125, -114.8, -114.8, -125]).data[0])
fig.add_trace(px.line_geo(lat = [55.75, 50.75, 50.75, 55.75, 55.75], lon = [-123, -123, -118, -118, -123]).data[0])
fig.add_trace(px.line_geo(lat = [58, 51, 51, 58, 58], lon = [-130.5, -130.5, -123.75, -123.75, -130.5]).data[0])
fig.update_layout(coloraxis_colorbar_title_text = 'Region')
fig.update_geos(fitbounds="locations",
                resolution=50,
                showcoastlines=True, coastlinecolor="RebeccaPurple",
                showland=True, landcolor="White",
                showocean=True, oceancolor="LightBlue",
                projection_type="conic equal area")
fig.update_geos(scope="north america",
                showcountries=True, countrycolor="Black",
                showsubunits=True, subunitcolor="Blue")
fig.write_image('era_regions.png', scale = 5)



