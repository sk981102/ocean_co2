import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def df_to_xarray(df_in=None):
    '''
    df_to_xarray(df_in) converts dataframe to dataset
        this makes a monthly 1x1 skeleton dataframe already
        time, lat, lon need to be in the dataframe
    !! this take 4 minutes !!
    example
    ==========
    ds = df_to_xarray(df_in = df[['time','lat','lon','sst']])
    '''
    # to make date in attributes
    from datetime import date
    # Make skeleton 
    dates = pd.date_range(start=f'1982-01-01', end=f'2018-12-01',freq='MS') + np.timedelta64(14, 'D')
    ds_skeleton = xr.Dataset({'lon':np.arange(0.5, 360, 1), 
                              'lat':np.arange(-89.5, 90, 1),
                              'time':dates})    
    # make dataframe
    skeleton = ds_skeleton.to_dataframe().reset_index()[['time','lat','lon']]
    # Merge predictions with df_all dataframe
    df_out = skeleton.merge(df_in, how = 'left', on = ['time','lat','lon'])
    # convert to xarray dataset
    # old way to `dimt, = ds_skeleton.time.shape` ect. to get dimensions
    # then reshape  `df_out.values.reshape(dim_lat, dim_lon, dim_time)`
    # finally create a custom dataset
    df_out.set_index(['time', 'lat','lon'], inplace=True)
    ds = df_out.to_xarray()
    #ds['sst'].attrs['units'] = 'uatm'
    return ds

def read_xarray(dir_name=""):
    '''
     read_xarray(dir)name) opens data and returns data in xarray format for each feature
    '''
    chl = xr.open_dataset(f'{dir_name}/Chl_2D_mon_CESM001_1x1_198201-201701.nc')

mld = xr.open_dataset(f'{dir_name}/MLD_2D_mon_CESM001_1x1_198201-201701.nc')

sss = xr.open_dataset(f'{dir_name}/SSS_2D_mon_CESM001_1x1_198201-201701.nc')

sst = xr.open_dataset(f'{dir_name}/SST_2D_mon_CESM001_1x1_198201-201701.nc')

u10 = xr.open_dataset(f'{dir_name}/U10_2D_mon_CESM001_1x1_198201-201701.nc')

fg_co2= xr.open_dataset(f'{dir_name}/FG-CO2_2D_mon_CESM001_1x1_198201-201701.nc')

xco2 = xr.open_dataset(f'{dir_name}/XCO2_1D_mon_CESM001_native_198201-201701.nc')

icefrac = xr.open_dataset(f'{dir_name}/iceFrac_2D_mon_CESM001_1x1_198201-201701.nc')

patm = xr.open_dataset(f'{dir_name}/pATM_2D_mon_CESM001_1x1_198201-201701.nc')

pco2 = xr.open_dataset(f'{dir_name}/pCO2_2D_mon_CESM001_1x1_198201-201701.nc')
    
    return chl,mld,sss,sst,u10,fg_co2,xco2,ice_frac,patm,pco2

