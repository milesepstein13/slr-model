import numpy as np
import pandas as pd
import cdsapi
import os


# setup cds api
cds = cdsapi.Client()
filename = "geopotentials"
print("requesting " + filename)
cds.retrieve('reanalysis-era5-single-levels', {
    "variable": ['geopotential'],
    "product_type": 'reanalysis',
    "date": '2021-12-31',
    "time": '00:00',
    "format": 'netcdf',
    "area": [58, -130.5, 48.25, -115]
}, filename+'.nc')