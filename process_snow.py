import numpy as np
import pandas as pd
import os

# writes start and endtimes in file of stations (assuming station files are in place from manual downloads)

# ORDER: 1

# download stations data
data = pd.read_csv("data/stations.csv")   
stations = pd.DataFrame(data)

pc_filestart = "DataSetExport-PC.Working@"
sd_filestart = "DataSetExport-SD.Working@"

# for each station
for i, station in stations.iterrows():
    if (station['Downloaded'] == True): # do for each usable station


        print(i)

        # download pc data
        pc_filename = pc_filestart + station.LCTN_ID
        for file in os.listdir("data/snow/"):
            if file.startswith(pc_filename): # file is name of each pc data file
                pc_full_filename = file
        
        data = pd.read_csv("data/snow/" + pc_full_filename, skiprows=2) 
        pc_data = pd.DataFrame(data)

        # Set start and end times
        pc_starttime = pc_data['Timestamp (UTC)'][0]
        stations.at[i,'PC Start'] = pc_starttime
        pc_endtime = pc_data['Timestamp (UTC)'].iloc[-1]
        stations.at[i,'PC End'] = pc_endtime
        
        # download sd data
        sd_filename = sd_filestart + station.LCTN_ID
        for file in os.listdir("data/snow/"):
            if file.startswith(sd_filename): # file is name of each psd data file
                sd_full_filename = file
        
        data = pd.read_csv("data/snow/" + sd_full_filename, skiprows=2) 
        sd_data = pd.DataFrame(data)

        # Set start and end times
        sd_starttime = sd_data['Timestamp (UTC)'][0]
        stations.at[i,'SD Start'] = sd_starttime
        sd_endtime = sd_data['Timestamp (UTC)'].iloc[-1]
        stations.at[i,'SD End'] = sd_endtime

output_path = "data/stations_new.csv"
print("writing to " + output_path)
# save stations to csv

stations.to_csv(output_path)