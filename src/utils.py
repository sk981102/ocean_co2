import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel

### Miscellaneous - might delete later ###
def un_standardized_mse(val):
  val=np.sqrt(val)*pco2.pCO2.data.max()
  return val

### FOR SEQUENTIAL + VISION

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

### FOR VISION ###
def process_xco2(xco2):
    xco2_images=[]

    for i in xco2:
        tmp=(np.repeat(i,180*360)).reshape(180,-1)
        xco2_images.append(tmp)
    
    return xco2_images

def plot_image(image):
    '''
    plots image of the map in black and white
    '''
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def convert_nan(arr):
    '''
    converts NaN values, which indicate coordinates into 0
    '''
    nans=np.isnan(arr)
    arr[nans]=0
    return arr

def add_dimension(arr):
    '''
    adds additional dimension to feed into cnn
    '''
    images=np.expand_dims(arr, axis=3)
    return images

def scale_image(arr):
    '''
    standardizing image values
    '''
    min_val=arr.min()
    arr=arr+abs(min_val)
    max_val=arr.max()
    arr=arr/max_val*255
    return arr
  
def preprocess_image(data,xco2=False,pco2=False):
    if xco2:
        return add_dimension(process_xco2(data))
    if pco2:
        return add_dimension(convert_nan(data))
    else:
        return add_dimension(scale_image(convert_nan(data))/255.0)




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

    return chl,mld,sss,sst,u10,fg_co2,xco2,icefrac,patm,pco2

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

def network_mask():
    '''network_mask
    This masks out regions in the 
    NCEP land-sea mask (https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html)
    to define the open ocean. Regions removed include:
    - Coast : defined by sobel filter
    - Batymetry less than 100m
    - Arctic ocean : defined as North of 79N
    - Hudson Bay
    - caspian sea, black sea, mediterranean sea, baltic sea, Java sea, Red sea
    '''
    ### Load obs directory
    dir_obs = '/local/data/artemis/observations'
    
    ### topography
    ds_topo = xr.open_dataset(f'{dir_obs}/GEBCO_2014/processed/GEBCO_2014_1x1_global.nc')
    ds_topo = ds_topo.roll(lon=180, roll_coords='lon')
    ds_topo['lon'] = np.arange(0.5, 360, 1)

    ### Loads grids
    # land-sea mask
    # land=0, sea=1
    ds_lsmask = xr.open_dataset(f'{dir_obs}/masks/originals/lsmask.nc').sortby('lat').squeeze().drop('time')
    data = ds_lsmask['mask'].where(ds_lsmask['mask']==1)
    ### Define Latitude and Longitude
    lon = ds_lsmask['lon']
    lat = ds_lsmask['lat']
    
    ### Remove coastal points, defined by sobel edge detection
    coast = (sobel(ds_lsmask['mask'])>0)
    data = data.where(coast==0)
    
    ### Remove shallow sea, less than 100m
    ### This picks out the Solomon islands and Somoa
    data = data.where(ds_topo['Height']<-100)
    
    ### remove arctic
    data = data.where(~((lat>79)))
    data = data.where(~((lat>67) & (lat<80) & (lon>20) & (lon<180)))
    data = data.where(~((lat>67) & (lat<80) & (lon>-180+360) & (lon<-100+360)))

    ### remove caspian sea, black sea, mediterranean sea, and baltic sea
    data = data.where(~((lat>24) & (lat<70) & (lon>14) & (lon<70)))
    
    ### remove hudson bay
    data = data.where(~((lat>50) & (lat<70) & (lon>-100+360) & (lon<-70+360)))
    data = data.where(~((lat>70) & (lat<80) & (lon>-130+360) & (lon<-80+360)))
    
    ### Remove Red sea
    data = data.where(~((lat>10) & (lat<25) & (lon>10) & (lon<45)))
    data = data.where(~((lat>20) & (lat<50) & (lon>0) & (lon<20)))
    
    return data
