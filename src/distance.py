import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import df_to_xarray,read_xarray

def coastline_dist_lat(m):
    height, width = m.shape[0], m.shape[1]
    output = np.zeros(shape = (height,width))
    
    for h in range(height):
        continents = np.sort(np.argwhere(np.isnan(m[h])))
        if len(continents) == 0:
            output[h] = np.ones(width)
        else:
            cur_shore = continents[0]
            ind=0
            
        for w in range(width):
            if w not in continents[1:]:
                if len(continents)-2 > ind:
                    before, after = continents[ind], continents[ind+1]
                    if after < before:
                        cur_shore = after
                        ind +=1
                    else:
                        cur_shore = before
                output[h][w] = abs(w-cur_shore)
            else:
                ind += 1
                cur_shore = continents[ind]
    return output

def coastline_dist_lon(m):
    height, width = m.shape[0], m.shape[1]
    output = np.zeros(shape = (height,width))
    
    for w in range(width):
        continents = np.sort(np.argwhere(np.isnan(m[:,w])))
        if len(continents) == 0:
            output[w] = np.ones(height)
        else:
            cur_shore = continents[0]
            ind=0
            
        for h in range(height):
            if h not in continents[1:]:
                if len(continents)-2 > ind:
                    before, after = continents[ind], continents[ind+1]
                    if after<before:
                        cur_shore = after
                        ind +=1
                    else:
                        cur_shore = before
                output[h][w] = abs(w-cur_shore)
            else:
                ind += 1
                cur_shore = continents[ind]
    return output