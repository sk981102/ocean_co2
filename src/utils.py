import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorflow.keras import backend as K


def custom_rmse(y_true, y_pred):
    """
    custom_rmse(y_true, y_pred)
    calculates root square mean value with focusing only on the ocean
    """
    y_pred = y_pred[(y_true != 0) & (y_true != 0.0)]
    y_true = y_true[(y_true != 0) & (y_true != 0.0)]
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    return K.sqrt(K.mean(tf.math.squared_difference(y_pred, y_true),axis= -1))


def eliminate_zero_pco2(pco2,socat=True):
    if socat:
        tmp=np.array(pco2.pCO2_socat.data)
    else:
        tmp=np.array(pco2.pCO2.data)
        
    ind=[]
    
    for i in range(421):
        ind.append(np.nanmax(tmp[i]) != 0)
    
    return ind,tmp[ind]

def inverse_scale_image(arr, df):
    """
    inverse_scale_image(arr, df):
    - inverses the pco2 scaling
    """
    
    old_min = np.nanmin(df)
    old_max = np.nanmax(df)

    output = arr*(old_max-old_min)/255+old_min
    return output

def inverse_scale_image_nfp(arr, df):
    """
    inverse_scale_image(arr, df):
    - inverses the pco2 scaling
    """
    
    old_min = np.nanmin(df)
    old_max = np.nanmax(df)

    y_pred = arr*(old_max-old_min)/255+old_min
    
    tmp=np.nan_to_num(pco2.pCO2.data[X_index][1:])
    y_true=np.expand_dims(tmp,axis=4)
    y_pred[y_true==0]=0
    return y_true,y_pred

def get_point_prediction(pred,lon,lan):
    pco2_value = pred[lan][lon]
    return pco2_value


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
    try:
        print("works")
        df_out = skeleton.merge(df_in, how = 'left', on = ['time','lat','lon'])
    except:
        print("maybe this")
        con=[skeleton,df_in]
        df_out=pd.concat(con)
        
    # convert to xarray dataset
    # old way to `dimt, = ds_skeleton.time.shape` ect. to get dimensions
    # then reshape  `df_out.values.reshape(dim_lat, dim_lon, dim_time)`
    # finally create a custom dataset
    df_out.set_index(['time', 'lat','lon'], inplace=True)
    ds = df_out.to_xarray()
    #ds['sst'].attrs['units'] = 'uatm'
    return ds

def read_xarray(dir_name="",num="001",mpi=False,can=False):
    '''
     read_xarray(dir)name) opens data and returns data in xarray format for each feature
    '''
    date="198201-201701"
    file_type = "CESM"
    if mpi:
        file_type ="MPI006"
        num=""
    elif can:
        file_type = "CanESM2r1r10"
        num=""
        date="198201-201712"
        
    
    chl = xr.open_dataset(f'{dir_name}/Chl_2D_mon_{file_type}{num}_1x1_{date}.nc')

    mld = xr.open_dataset(f'{dir_name}/MLD_2D_mon_{file_type}{num}_1x1_{date}.nc')

    sss = xr.open_dataset(f'{dir_name}/SSS_2D_mon_{file_type}{num}_1x1_{date}.nc')

    sst = xr.open_dataset(f'{dir_name}/SST_2D_mon_{file_type}{num}_1x1_{date}.nc')

    u10 = xr.open_dataset(f'{dir_name}/U10_2D_mon_{file_type}{num}_1x1_{date}.nc')

    xco2 = xr.open_dataset(f'../../data/data1/XCO2_1D_mon_CESM001_native_198201-201701.nc')

    icefrac = xr.open_dataset(f'{dir_name}/iceFrac_2D_mon_{file_type}{num}_1x1_{date}.nc')

    patm = xr.open_dataset(f'{dir_name}/pATM_2D_mon_{file_type}{num}_1x1_{date}.nc')

    pco2 = xr.open_dataset(f'{dir_name}/pCO2_2D_mon_{file_type}{num}_1x1_{date}.nc')

    return chl,mld,sss,sst,u10,xco2,icefrac,patm,pco2



def repeat_lat_and_lon(ds=None):
    lon = np.arange(0.5,360,1)
    lat = np.arange(-89.5,90,1)
    ds_bc = xr.DataArray(np.zeros([len(lon),len(lat)]), coords=[('lon', lon),('lat', lat)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_lon(ds=None):
    lon = np.arange(0.5,360,1)
    ds_bc = xr.DataArray(np.zeros([len(lon)]), coords=[('lon', lon)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_lat(ds=None):
    lat = np.arange(-89.5,90,1)
    ds_bc = xr.DataArray(np.zeros([len(lat)]), coords=[('lat', lat)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_time(ds=None, dates=None):
    ''' dates needs to be a pandas date_range
    Example
    dates = pd.date_range(start='1982-01-01T00:00:00.000000000', 
                      end='2017-12-01T00:00:00.000000000',freq='MS')+ np.timedelta64(14, 'D')
    '''
    ds_bc = xr.DataArray(np.zeros([len(dates)]), coords=[('time', dates)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def repeat_time_and_lon(ds=None, dates=None):
    ''' dates needs to be a pandas date_range
    Example
    dates = pd.date_range(start='1998-01-01T00:00:00.000000000', 
                      end='2017-12-01T00:00:00.000000000',freq='MS')+ np.timedelta64(14, 'D')
    '''
    lon = np.arange(0.5,360,1)
    ds_bc = xr.DataArray(np.zeros([len(dates), len(lon), ]), coords=[('time', dates),('lon', lon)])
    ds_data, ds_mask = xr.broadcast(ds, ds_bc)
    return ds_data

def transform_doy(obj, dim='time'):
    '''
    transform_doy(ds, dim='time')
    transform DOY into repeating cycles
    
    reference
    ==========
    Gregor et al. 2019 
    '''
    obj['T0'] = np.cos(obj[f'{dim}.dayofyear'] * 2 * np.pi / 365)
    obj['T1'] = np.sin(obj[f'{dim}.dayofyear'] * 2 * np.pi / 365)
    return obj[['T0','T1']]

def compute_n_vector(obj, dim_lon='lon', dim_lat='lat'):
    '''
    compute_n_vector(ds,dim_lon='lon', dim_lat='lat')
    calculate n-vector from lat/lon
    
    reference
    ==========
    Gregor et al. 2019 
    '''
    ### convert lat/lon to radians
    obj['lat_rad'], obj['lon_rad'] = np.radians(obj[dim_lat]), np.radians(obj[dim_lon])

    ### Calculate n-vector
    obj['A'], obj['B'], obj['C'] = np.sin(obj['lat_rad']), \
                            np.sin(obj['lon_rad'])*np.cos(obj['lat_rad']), \
                            -np.cos(obj['lon_rad'])*np.cos(obj['lat_rad'])
    return obj[['A','B','C']]



# def network_mask():
#     '''network_mask
#     This masks out regions in the 
#     NCEP land-sea mask (https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html)
#     to define the open ocean. Regions removed include:
#     - Coast : defined by sobel filter
#     - Batymetry less than 100m
#     - Arctic ocean : defined as North of 79N
#     - Hudson Bay
#     - caspian sea, black sea, mediterranean sea, baltic sea, Java sea, Red sea
#     '''
#     ### Load obs directory
#     dir_obs = '/local/data/artemis/observations'
    
#     ### topography
#     ds_topo = xr.open_dataset(f'{dir_obs}/GEBCO_2014/processed/GEBCO_2014_1x1_global.nc')
#     ds_topo = ds_topo.roll(lon=180, roll_coords='lon')
#     ds_topo['lon'] = np.arange(0.5, 360, 1)

#     ### Loads grids
#     # land-sea mask
#     # land=0, sea=1
#     ds_lsmask = xr.open_dataset(f'{dir_obs}/masks/originals/lsmask.nc').sortby('lat').squeeze().drop('time')
#     data = ds_lsmask['mask'].where(ds_lsmask['mask']==1)
#     ### Define Latitude and Longitude
#     lon = ds_lsmask['lon']
#     lat = ds_lsmask['lat']
    
#     ### Remove coastal points, defined by sobel edge detection
#     coast = (sobel(ds_lsmask['mask'])>0)
#     data = data.where(coast==0)
    
#     ### Remove shallow sea, less than 100m
#     ### This picks out the Solomon islands and Somoa
#     data = data.where(ds_topo['Height']<-100)
    
#     ### remove arctic
#     data = data.where(~((lat>79)))
#     data = data.where(~((lat>67) & (lat<80) & (lon>20) & (lon<180)))
#     data = data.where(~((lat>67) & (lat<80) & (lon>-180+360) & (lon<-100+360)))

#     ### remove caspian sea, black sea, mediterranean sea, and baltic sea
#     data = data.where(~((lat>24) & (lat<70) & (lon>14) & (lon<70)))
    
#     ### remove hudson bay
#     data = data.where(~((lat>50) & (lat<70) & (lon>-100+360) & (lon<-70+360)))
#     data = data.where(~((lat>70) & (lat<80) & (lon>-130+360) & (lon<-80+360)))
    
#     ### Remove Red sea
#     data = data.where(~((lat>10) & (lat<25) & (lon>10) & (lon<45)))
#     data = data.where(~((lat>20) & (lat<50) & (lon>0) & (lon<20)))
    
#     return data
