#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:16:03 2019

@author: deepak
"""

"""Script to select the stations from the entire set
    This will generate the train, test and the pool set of stations"""
    
import pandas as pd
import numpy as np
from copy import deepcopy

df = pd.read_csv('../../Datasets/Updated_Delhi_Scaled.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

station_list = list(df['Station'].unique())


station_numbering = {}
for i in range(len(station_list)):
    station_numbering[station_list[i]] = 'S'+str(i+1)
    
for i in range(len(station_list)):
    test_set = [station_list[i]]
    new_station_list = deepcopy(station_list)
    new_station_list.remove(station_list[i])
    train_set = new_station_list[:4]
    pool_set = new_station_list[4:]
    
