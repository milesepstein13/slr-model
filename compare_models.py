import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
import scipy as sp
from scipy import stats
import plotly.express as px
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.calc import dewpoint_from_relative_humidity
from metpy.units import units

np.random.seed(0)

# to get only validation examples from dataset
def get_validation(dataset):
    values = dataset.values
    np.random.shuffle(values)
    #print(values.shape)
    #print(dataset.shape)
    dataset = pd.DataFrame(values[:, 1:], columns =dataset.columns[1:])
    #print(dataset)
    n = dataset.shape[0]
    split = int(8*n/10)
    dataset = dataset.iloc[split + 1:, :]
    return dataset

base_name = 'data/datasets/test_data_6_modified'

# load in validation datasets (each dataset contains different variables)
print("Loading datasets")
dataset_names = ["", "_small", "_without_derived", "_without_meta", "_without_radiation", "_without_surface", "_without_temperature", "_without_upper", "_without_water", "_without_wind"]
validation_datasets = dict()
for name in dataset_names:
    validation_datasets[name] = get_validation(pd.read_csv(base_name + name + '.csv'))


lookup = pd.read_csv('nws_lookup.csv')

def predict_ten(dataset):
    print("Predicting with 10")
    return 10*np.ones(dataset.shape[0])

def predict_mean(dataset):
    print("Predicting with mean")
    return np.ones(dataset.shape[0]) * np.mean(dataset.SLR)

def predict_station_mean(dataset):
    print("Predicting with Station Mean")
    # distinguishes stations by latitude (assumes two dont have same lat)
    lats = list(set(dataset.latitude))
    slrs = np.zeros(len(lats))
    for i in range(len(lats)):
        slrs[i] = np.mean(dataset.SLR[dataset.latitude == lats[i]])
    slr_dict = dict(zip(lats, slrs))
    ret = []
    for i in range(dataset.shape[0]):
        ret.append(slr_dict[dataset.latitude.iloc[i]])
    return ret

def tempmax(row):
    # returns max temp in lowest 500 mb
    max = row['2m_temperature']
    surface_pressure = row.surface_pressure/100
    pressure_levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200]
    tempstr = 'temperature_'
    for pressure in pressure_levels:
        if (pressure + 500 > surface_pressure) and (pressure < surface_pressure):
            st = tempstr + str(pressure)
            if row[st] > max:
                max = row[st]
    return max

def predict_kuchera(dataset):
    print("Predicting with Kuchera Method")
    ret = []
    for i in range(dataset.shape[0]):
        tmax = tempmax(dataset.iloc[i])
        if tmax > 271.16:
            pred = 12 + 2*(271.16 - tmax)
        else:
            pred = 12 + (271.16 - tmax)
        ret.append(pred)
    return ret

# only a lookup table, not used operationally
def predict_nws(dataset):
    ret = []
    depths = dataset.total_precipitation
    temps = dataset['2m_temperature']
    for i in range(dataset.shape[0]):
        SWE_inches = depths[i] * 39.37 * 6
        if SWE_inches < .008:
            SWE_inches = .01
        j = np.argmin(np.abs(lookup.SWE-SWE_inches))
        temp = temps[i]
        if temp > 274.261:
            ret.append(0)
        elif temp > 270.928:
            ret.append(lookup.loc[j]['28-34']/SWE_inches)
        elif temp > 266.483:
            ret.append(lookup.loc[j]['20-27']/SWE_inches)
        elif temp > 263.706:
            ret.append(lookup.loc[j]['15-19']/SWE_inches)
        elif temp > 260.928:
            ret.append(lookup.loc[j]['10-14']/SWE_inches)
        elif temp > 255.372:
            ret.append(lookup.loc[j]['0-9']/SWE_inches)
        elif temp > 244.261:
            ret.append(lookup.loc[j]['-1--20']/SWE_inches)
        else:
            ret.append(lookup.loc[j]['-21--40']/SWE_inches)
    return ret

def predict_dube(dataset):
    print("Predicting with Dube Method")
    # note: don't have ground temperature in dataset, so not full dube
    ret = []
    for i in range(dataset.shape[0]):
        data = dataset.iloc[i]
        if tempmax(data) > 273.15:
            ret.append(predict_dube_positive(data))
        else:
            ret.append(predict_dube_negative(data))
    return ret

def predict_dube_positive(data):
    return 7
    #TODO: implement

def predict_dube_negative(data):
    
    (tprim, tprimp) = tprimary(data)
    tsec = tsecondary(data, tprimp)
    sub = sublimation(data)
    acc = accretion(data)
    vmax = get_vmax(data) * 1.94384 #knots

    tprim = tprim - 273.15
    tsec = tsec - 273.15

    if (-5 <= tprim <= -3) and (-5 <= tsec <= -3):
        return(adjust_needles(vmax))
    elif (-18 <= tprim <= -12) and (-18 <= tsec <= -12):
        return(adjust_stars(acc, sub, vmax))
    elif (-18 <= tprim <= -12) and (tsec <= -5):
        return(adjust_mixed_stellar(acc, sub, vmax))
    elif (-18 <= tprim <= -12) and (tsec <= -5):
        return(adjust_dendrites(acc, sub, vmax))
    else:
        return(adjust_mixed(acc))
    
def adjust_needles(vmax):
    if vmax  > 25:
        return 10
    return 15

def adjust_stars(acc, sub, vmax):
    
    if acc:
        return 10
    if sub:
        if vmax > 15:
            return 10
        return 15
    if vmax < 5:
        return 25
    if vmax < 15:
        return 20
    if vmax < 25:
        return 15
    return 10


def adjust_mixed_stellar(acc, sub, vmax):
    if acc or sub or (vmax > 25):
        return 10
    return 15

def adjust_dendrites(acc, sub, vmax):
    if acc:
        return 7
    return 10


def adjust_mixed(acc):
    if acc:
        return 7
    return 10

def tprimary(data):
    pressures = get_pressures(data)[1:]
    rh = np.array(relative_humidities(data, pressures))
    str1 = 'vertical_velocity_' + str(pressures[0])
    vvs = data.loc[str1 : 'vertical_velocity_200']
    if np.max(rh) < .8:
        return (data['temperature_' + str(pressures[np.argmax(vvs)])], pressures[np.argmax(vvs)])
    else:
        vvmax = -np.inf
        for i in range(len(vvs)):
            if (vvs[i] > vvmax) and (rh[i] >= .8):
                vvmax = vvs[i]
                pmax = pressures[i]
        return (data['temperature_' + str(pmax)], pmax)

def get_pressures(data):
    pressures = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200]
    i = 0
    while True:
        if pressures[i] < data.surface_pressure/100:
            return pressures[i:]
        else:
            i = i+1

def relative_humidities(data, pressures):
    rh = []
    for p in pressures:
        tempstr = 'temperature_' + str(p)
        shstr = 'specific_humidity_' + str(p)
        rh.append(float(relative_humidity_from_specific_humidity(p * units.hPa, data[tempstr] * units.degK, data[shstr])))
    return rh

def dewpoints(data, pressures):
    dp = []
    for p in pressures:
        tempstr = 'temperature_' + str(p)
        shstr = 'specific_humidity_' + str(p)
        rh = float(relative_humidity_from_specific_humidity(p * units.hPa, data[tempstr] * units.degK, data[shstr]))
        temp = data['temperature_' + str(p)] - 273.15
        dew = float(str(dewpoint_from_relative_humidity(data[tempstr] * units.degK, rh * units.percent)).split(' ')[0])
        dp.append(temp - dew)
    return dp

def tsecondary(data, tprimp):
    pressures = get_pressures(data)
    i = np.where(np.array(pressures) == tprimp)[0][0]
    pressures = pressures[:i]
    rh = np.array(relative_humidities(data, pressures))
    for i in reversed(range(len(pressures))):
        if (rh[i] > .8) and (data['vertical_velocity_' + str(pressures[i])] < 0) and (data['temperature_' + str(pressures[i])] < 273.15):
            return data['temperature_' + str(pressures[i])]
    return data['temperature_' + str(tprimp)]   

def accretion(data):
    # TODO: Don't think I can check for supercooled with my dataset, using rain instead
    pressures = get_pressures(data)
    rh = np.array(relative_humidities(data, pressures))
    for p, i in zip(pressures, range(len(pressures))):
        if data['specific_rain_water_content_' + str(p)] > 0 and data['temperature_' + str(p)] > 263.15 and data['specific_rain_water_content_' + str(p)] < 273.15 and data['specific_rain_water_content_' + str(p)] > 0 and rh[i]>.95 and data['u_component_of_wind_' + str(p)] > 0 and data['v_component_of_wind_' + str(p)] > 0:
            return True
    return False

def sublimation(data):
    pressures = cloud_base_pressures(data)
    if pressures == []:
        return False
    rhs = np.array(relative_humidities(data, pressures))
    dews = np.array(dewpoints(data, pressures))
    if np.min(rhs) < .8:
        return True
    if np.max(dews) > 3:
        return True
    return False

def get_vmax(data):
    pressures = cloud_base_pressures(data)
    w1 = np.sqrt(data['10m_u_component_of_wind']**2 + data['10m_v_component_of_neutral_wind']**2)
    w2 = np.sqrt(data['100m_u_component_of_wind']**2 + data['100m_v_component_of_wind']**2)
    wind = max(w1, w2)
    for p in pressures:
        test = np.sqrt(data['u_component_of_wind_' + str(p)]**2 + data['v_component_of_wind_' + str(p)]**2)
        if test > wind:
            wind = test
    return wind

def cloud_base_pressures(data):
    pressures = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200]
    i = 0
    done = False
    while done == False:
        if pressures[i] < data.surface_pressure/100:
            j = i
            done = True
        else:
            i = i+1
    while i < len(pressures):
        p = pressures[i]
        tempstr = 'temperature_' + str(p)
        shstr = 'specific_humidity_' + str(p)
        rh = float(relative_humidity_from_specific_humidity(p * units.hPa, data[tempstr] * units.degK, data[shstr]))
        if rh > .95:
            return pressures[j:i]
        i = i+1
    return pressures[j:]



def predict_nn(dataset, num):
    model = keras.models.load_model("models/"+str(num))
    #print(dataset)
    X = dataset.to_numpy()[:, :-1]
    #print(X)
    return(model.predict(X))

# other prediction methods: others from studies

def get_station_nums(dataset):
    return stats.rankdata(dataset.latitude, method='dense')

def get_clusters(dataset):
    clusters = np.zeros(len(dataset.latitude))
    for i in range(len(dataset.latitude)):
        lat = dataset.latitude.iloc[i]
        lon = dataset.longitude.iloc[i]
        if (lon < -127 or (lon < -122.5 and lat < 54.5) or (lon < -120 and lat < 51)):
            clusters[i] = 1
        else:
            clusters[i] = 2
    #TODO: put 3 regions here?
    fig = px.scatter_geo(lat = dataset.latitude, lon = dataset.longitude, color = clusters, title = "Station Locations, Colored by Region")
    fig.update_geos(fitbounds="locations",
                    resolution=50,
                    showcoastlines=True, coastlinecolor="RebeccaPurple",
                    showland=True, landcolor="LightGreen",
                    showocean=True, oceancolor="LightBlue",
                    projection_type="conic equal area")
    fig.update_geos(scope="north america",
                    showcountries=True, countrycolor="Black",
                    showsubunits=True, subunitcolor="Blue")
    fig.write_image('regions.png')

    return clusters

print("Making predictions...")

predictions = pd.DataFrame().assign(recorded = validation_datasets[""].SLR, 
                                    ten = predict_ten(validation_datasets[""]), 
                                    meann = predict_mean(validation_datasets[""]),
                                    station_mean = predict_station_mean(validation_datasets[""]),
                                    kuchera = predict_kuchera(validation_datasets[""]),
                                    dube = predict_dube(validation_datasets[""]),
                                    nn_full = predict_nn(validation_datasets[""], 178),
                                    nn_small = predict_nn(validation_datasets["_small"], 179),
                                    nn_without_derived = predict_nn(validation_datasets["_without_derived"], 180),
                                    nn_without_meta = predict_nn(validation_datasets["_without_meta"], 181),
                                    nn_without_radiation = predict_nn(validation_datasets["_without_radiation"], 182),
                                    nn_without_surface = predict_nn(validation_datasets["_without_surface"], 183),
                                    nn_without_temperature = predict_nn(validation_datasets["_without_temperature"], 184),
                                    nn_without_upper = predict_nn(validation_datasets["_without_upper"], 185),
                                    nn_without_water = predict_nn(validation_datasets["_without_water"], 186),
                                    nn_without_wind = predict_nn(validation_datasets["_without_wind"], 187),
                                    station = get_station_nums(validation_datasets[""]),
                                    cluster = get_clusters(validation_datasets[""]))

print(predictions)

# there's a better way to do this, but too late now
SE = pd.DataFrame().assign(recorded = ((predictions.recorded-predictions.recorded)**2), 
                            ten = ((predictions.recorded-predictions.ten)**2), 
                            mean = ((predictions.recorded-predictions.meann)**2),
                            station_mean = ((predictions.recorded-predictions.station_mean)**2),
                            kuchera = ((predictions.recorded-predictions.kuchera)**2),
                            dube = ((predictions.recorded-predictions.dube)**2),
                            nn_full = ((predictions.recorded-predictions.nn_full)**2),
                            nn_small = ((predictions.recorded-predictions.nn_small)**2),
                            nn_without_derived = ((predictions.recorded-predictions.nn_without_derived)**2),
                            nn_without_meta = ((predictions.recorded-predictions.nn_without_meta)**2),
                            nn_without_radiation = ((predictions.recorded-predictions.nn_without_radiation)**2),
                            nn_without_surface = ((predictions.recorded-predictions.nn_without_surface)**2),
                            nn_without_temperature = ((predictions.recorded-predictions.nn_without_temperature)**2),
                            nn_without_upper = ((predictions.recorded-predictions.nn_without_upper)**2),
                            nn_without_water = ((predictions.recorded-predictions.nn_without_water)**2),
                            nn_without_wind = ((predictions.recorded-predictions.nn_without_wind)**2),
                            station = predictions.station,
                            cluster = predictions.cluster)

AE = pd.DataFrame().assign(recorded = (np.abs(predictions.recorded-predictions.recorded)), 
                            ten = (np.abs(predictions.recorded-predictions.ten)), 
                            mean = (np.abs(predictions.recorded-predictions.meann)),
                            station_mean = (np.abs(predictions.recorded-predictions.station_mean)),
                            kuchera = (np.abs(predictions.recorded-predictions.kuchera)),
                            dube = (np.abs(predictions.recorded-predictions.dube)),
                            nn_full = (np.abs(predictions.recorded-predictions.nn_full)),
                            nn_small = (np.abs(predictions.recorded-predictions.nn_small)),
                            nn_without_derived = (np.abs(predictions.recorded-predictions.nn_without_derived)),
                            nn_without_meta = (np.abs(predictions.recorded-predictions.nn_without_meta)),
                            nn_without_radiation = (np.abs(predictions.recorded-predictions.nn_without_radiation)),
                            nn_without_surface = (np.abs(predictions.recorded-predictions.nn_without_surface)),
                            nn_without_temperature = (np.abs(predictions.recorded-predictions.nn_without_temperature)),
                            nn_without_upper = (np.abs(predictions.recorded-predictions.nn_without_upper)),
                            nn_without_water = (np.abs(predictions.recorded-predictions.nn_without_water)),
                            nn_without_wind = (np.abs(predictions.recorded-predictions.nn_without_wind)),
                            station = predictions.station,
                            cluster = predictions.cluster)
print(SE)

#TODO: other error metrics

MSE = SE.mean()
print("Overall MSE: ")
print(MSE)

MAE = AE.mean()
print("Overall MAE: ")
print(MAE)

RMSE = np.sqrt(MSE)
print("overall RMSE:")
print(RMSE)


SE_stations = SE.groupby('station')
MSE_locations = SE_stations.mean()
MSE_locations['count'] = SE_stations.count().recorded
print("MSE Locations:")
print(MSE_locations)

RMSE_stations = np.sqrt(SE_stations.mean())
RMSE_stations['count'] = SE_stations.count().recorded


AE_stations = AE.groupby('station')
MAE_stations = AE_stations.mean()
MAE_stations['count'] = AE_stations.count().recorded



SE_clusters = SE.groupby('cluster')
MSE_clusters = SE_clusters.mean()
MSE_clusters['count'] = SE_clusters.count().recorded
print("MSE Clusters:")
print(MSE_clusters)

RMSE_clusters = np.sqrt(SE_clusters.mean())
RMSE_clusters['count'] = SE_clusters.count().recorded

AE_clusters = AE.groupby('cluster')
MAE_clusters = AE_clusters.mean()
MAE_clusters['count'] = AE_clusters.count().recorded


print("Saving data...")
# save data
predictions.to_csv('outputs/predictions.csv')
MSE.to_csv('outputs/overall_mse.csv')
MSE_locations.to_csv('outputs/station_mse.csv')
MSE_clusters.to_csv('outputs/cluster_mse.csv')

RMSE.to_csv('outputs/overall_rmse.csv')
RMSE_stations.to_csv('outputs/station_rmse.csv')
RMSE_clusters.to_csv('outputs/cluster_rmse.csv')

MAE.to_csv('outputs/overall_mae.csv')
MAE_stations.to_csv('outputs/station_mae.csv')
MAE_clusters.to_csv('outputs/cluster_mae.csv')


# put tables to latex:
#print(MSE.to_latex())
#print(MSE_locations.to_latex())
#print(MSE_clusters.to_latex())

# make figures
# not sure what figures to make? Should put station-by-station data on a map. Ask rosie what

def make_maps(value_locations, lats, lons, errortype, short_errortype):
    for column in value_locations:
        if (column != 'recorded') and (column != 'cluster') and (column != 'count'):
            #plt.scatter(lons, lats, c = value_locations[column], s=value_locations['count'], cmap='rainbow')

            #plt.title("Mean Test " + errortype + " With " + column + " Method for SLR Prediction")
            #plt.colorbar()
            #plt.xlabel("lon")
            #plt.ylabel("lat")
            
            #plt.savefig('outputs/figs/' + short_errortype + column + '.png')
            #plt.clf()
            print("plotting")

            fig = px.scatter_geo(lat = lats, lon = lons, color = value_locations[column], size = value_locations['count'], title = "Mean Test " + errortype + " With " + column + " Method for SLR Prediction")
            fig.update_geos(fitbounds="locations",
                            resolution=50,
                            showcoastlines=True, coastlinecolor="RebeccaPurple",
                            showland=True, landcolor="LightGreen",
                            showocean=True, oceancolor="LightBlue",
                            projection_type="conic equal area")
            fig.update_geos(scope="north america",
                            showcountries=True, countrycolor="Black",
                            showsubunits=True, subunitcolor="Blue")
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


(lats, lons) = get_lat_lon(validation_datasets[''])

# do for each
make_maps(MSE_locations, lats, lons, 'mean squared error', '_mse_')
make_maps(RMSE_stations, lats, lons, 'root mean squared error', '_rmse_')
make_maps(MAE_stations, lats, lons, 'mean absolute error', '_mae_')