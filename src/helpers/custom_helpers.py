# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:21:49 2018

@author: silus
"""
import numpy as np


def normalize_data(data):
    
    data[data == -999] = np.nan
    
    features_mean = np.nanmean(data, axis=0)
    features_stdev = np.nanstd(data, axis=0)

    tmp = (data-features_mean)/features_stdev
    #tmp[np.isnan(tmp)] = -999

    return tmp, features_mean, features_stdev
        
    
    