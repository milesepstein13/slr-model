import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

methoddict = {'kuchera': 'Kuchera Method',
                  'dube': "Dube Method",
                  'gpt': 'ChatGPT Method',
                  'mean': 'Predicting Overall Training Mean',
                  'station_mean': 'Predicting Station Training Mean',
                  'ten': 'Predicting Ten',
                  'nn_full': 'Neural Network Trained on Full Dataset',
                  'nn_small': 'Neural Network Trained on Thinned Dataset',
                  'nn_without_derived': 'Neural Network Trained on Dataset Without Derived Variables',
                  'nn_without_meta': 'Neural Network Trained on Dataset Without Metadata',
                  'nn_without_radiation': 'Neural Network Trained on Dataset Without Radiation',
                  'nn_without_surface': 'Neural Network Trained on Dataset Without Surface Variables',
                  'nn_without_temperature': 'Neural Network Trained on Dataset Without Temperatures',
                  'nn_without_upper': 'Neural Network Trained on Dataset Without Upper-Air Variables',
                  'nn_without_water': 'Neural Network Trained on Dataset Without Atmospheric Water Content',
                  'nn_without_wind': 'Neural Network Trained on Dataset Without Wind',
                  'nn_surface_temperature': 'Neural Network Trained on Dataset with only Surface Temperatures',
                  'nn_surface_temp_meta': 'Neural Network Trained on Dataset with only Surface Temperatures and Metadata',
                  'nn_temp_humidity': 'Neural Network Trained on Dataset with only Temperatues and Humidities',
                  'nn_temp_humidity_wind': 'Neural Network Trained on Dataset with only Temperatures, Humidities, and Wind',
                  'nn_temp_humidity_wind_meta': 'Neural Network Trained on Dataset with only Temperatures, Humidities, Wind, and Metadata'
                  }

def get_validation(dataset):
    values = dataset.values
    np.random.seed(0)
    np.random.shuffle(values)
    #print(values.shape)
    #print(dataset.shape)
    dataset = pd.DataFrame(values[:, 1:], columns =dataset.columns[1:])
    #print(dataset)
    n = dataset.shape[0]
    split = int(8*n/10)
    dataset = dataset.iloc[split + 1:, :]
    print(dataset)
    return dataset

dataset = get_validation(pd.read_csv('data/datasets/test_data_6_modified.csv'))

print("making bar charts")

MSE = pd.read_csv('outputs/overall_mse.csv')
MSE_locations = pd.read_csv('outputs/station_mse.csv')
MSE_clusters = pd.read_csv('outputs/cluster_mse.csv')

RMSE = pd.read_csv('outputs/overall_rmse.csv')
RMSE_stations = pd.read_csv('outputs/station_rmse.csv')
RMSE_clusters = pd.read_csv('outputs/cluster_rmse.csv')

MAE = pd.read_csv('outputs/overall_mae.csv')
MAE_stations = pd.read_csv('outputs/station_mae.csv')
MAE_clusters = pd.read_csv('outputs/cluster_mae.csv')


MAE = MAE.iloc[1:-2, :]
print(MAE)
plt.rcParams.update({'font.size': 22})

fig = MAE.plot.bar(legend = False, title = "Mean Absolute Test Error in SLR with Different Prediction Methods")
fig.set_xlabel('Method Number')
fig.set_ylabel('Mean Absolute Test Error')
fig.figure.savefig('outputs/figs/mae.png')

RMSE = RMSE.iloc[1:-2, :]
fig = RMSE.plot.bar(legend = False, title = "Root Mean Square Test Error in SLR with Different Prediction Methods")
fig.set_xlabel('Method Number')
fig.set_ylabel('Root Mean Square Test Error')
fig.figure.savefig('outputs/figs/rmse.png')


MAE_clusters = MAE_clusters.drop(columns=['count', 'station'])
fig = MAE_clusters.plot.bar()
fig.figure.savefig('outputs/figs/mae_clusters.png')

RMSE_clusters = RMSE_clusters.drop(columns=['count', 'station'])
fig = RMSE_clusters.plot.bar()
fig.figure.savefig('outputs/figs/rmse_clusters.png')


def make_maps(value_locations, lats, lons, errortype, short_errortype):
    methoddict = {'kuchera': 'Kuchera Method',
                  'dube': "Dube Method",
                  'gpt': 'ChatGPT Method',
                  'mean': 'Predicting Overall Training Mean',
                  'station_mean': 'Predicting Station Training Mean',
                  'ten': 'Predicting Ten',
                  'nn_full': 'Neural Network Trained on Full Dataset',
                  'nn_small': 'Neural Network Trained on Thinned Dataset',
                  'nn_without_derived': 'Neural Network Trained on Dataset Without Derived Variables',
                  'nn_without_meta': 'Neural Network Trained on Dataset Without Metadata',
                  'nn_without_radiation': 'Neural Network Trained on Dataset Without Radiation',
                  'nn_without_surface': 'Neural Network Trained on Dataset Without Surface Variables',
                  'nn_without_temperature': 'Neural Network Trained on Dataset Without Temperatures',
                  'nn_without_upper': 'Neural Network Trained on Dataset Without Upper-Air Variables',
                  'nn_without_water': 'Neural Network Trained on Dataset Without Atmospheric Water Content',
                  'nn_without_wind': 'Neural Network Trained on Dataset Without Wind',
                  'nn_surface_temperature': 'Neural Network Trained on Dataset with only Surface Temperatures',
                  'nn_surface_temp_meta': 'Neural Network Trained on Dataset with only Surface Temperatures and Metadata',
                  'nn_temp_humidity': 'Neural Network Trained on Dataset with only Temperatues and Humidities',
                  'nn_temp_humidity_wind': 'Neural Network Trained on Dataset with only Temperatures, Humidities, and Wind',
                  'nn_temp_humidity_wind_meta': 'Neural Network Trained on Dataset with only Temperatures, Humidities, Wind, and Metadata'
                  }
    print(value_locations)
    for column in value_locations:
        print(column)
    for column in value_locations:
        if (column != 'recorded') and (column != 'cluster') and (column != 'count') and (column != 'station'):
            #plt.scatter(lons, lats, c = value_locations[column], s=value_locations['count'], cmap='rainbow')

            #plt.title("Mean Test " + errortype + " With " + column + " Method for SLR Prediction")
            #plt.colorbar()
            #plt.xlabel("lon")
            #plt.ylabel("lat")
            
            #plt.savefig('outputs/figs/' + short_errortype + column + '.png')
            #plt.clf()
            print("plotting")
            if short_errortype == 'mae_':
                range = (1, 7)
            elif short_errortype == 'mse_':
                range = (10, 60)
            else:
                range = (3, 8)

            print(column)
            fig = px.scatter_geo(lat = lats, lon = lons, color = value_locations[column], size = value_locations['count'], range_color=range, color_continuous_scale=px.colors.sequential.Turbo)
            fig.update_geos(fitbounds="locations",
                            resolution=50,
                            showcoastlines=True, coastlinecolor="RebeccaPurple",
                            showland=True, landcolor="White",
                            showocean=True, oceancolor="LightBlue",
                            projection_type="conic equal area")
            fig.update_geos(scope="north america",
                            showcountries=True, countrycolor="Black",
                            showsubunits=True, subunitcolor="Blue")
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=errortype,
                    len = .8
                )
            )
            # no description in image, just make caption manually
            fig.write_image('outputs/figs/' + short_errortype + column + '.png')
            
        

def get_lat_lon(dataset):
    lats = np.unique(dataset.latitude)
    lats_order = list(dict.fromkeys(dataset.latitude))
    lons_order = list(dict.fromkeys(dataset.longitude))
    dic = dict(zip(lats_order, lons_order))
    lons = []
    for lat in lats:
        lons = lons + [dic[lat]]
    return (lats, lons)


(lats, lons) = get_lat_lon(dataset)


# do for each
make_maps(MSE_locations, lats, lons, 'Mean Squared Error', 'mse_')
make_maps(RMSE_stations, lats, lons, 'Root Mean Squared Error', 'rmse_')
make_maps(MAE_stations, lats, lons, 'Mean<br>Absolute<br>Error', 'mae_')