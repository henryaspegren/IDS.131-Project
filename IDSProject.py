#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:44:03 2017

@author: kai
"""
import datetime
import matplotlib
#==============================================================================
# matplotlib.use('Agg')
#==============================================================================
import random
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt 
import matplotlib.transforms as tfms
import seaborn as sns
import math
from matplotlib.path import Path
import matplotlib.patches as patches
import copy
import cmocean
import networkx as nx
import pickle as pkl
sns.set(color_codes=True)
sns.set_style("whitegrid")
import json

#%%

def getCooffending():
    co_offending_table = pd.read_csv('./Cooffending.csv')
    
    # Remove duplicate rows
    co_offending_table.drop_duplicates(inplace=True)
    
    # Format the date column as a python datetime object
    co_offending_table['Date'] = co_offending_table.Date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
    co_offending_table['year']=co_offending_table.Date.dt.year
    co_offending_table['month']=co_offending_table.Date.dt.month
    # Add a column for the number of arrests of each offender
    co_offending_table['ArrestCount'] = co_offending_table.groupby('NoUnique')['NoUnique'].transform('count')
    
    # Get the right datatype 
    co_offending_table.SeqE = co_offending_table.SeqE.astype('category')
    co_offending_table.SEXE = co_offending_table.SEXE.astype('category')
    co_offending_table.NCD1 = co_offending_table.NCD1.astype('category')
    co_offending_table.NCD2 = co_offending_table.NCD2.astype('category')
    co_offending_table.NCD3 = co_offending_table.NCD3.astype('category')
    co_offending_table.NCD4 = co_offending_table.NCD4.astype('category')
    co_offending_table.MUN = co_offending_table.MUN.astype('category')
    co_offending_table.ED1 = co_offending_table.ED1.astype('category')
    return co_offending_table

#training_data = build_table_of_first_two_arrests(co_offending_table)

# or read from csv

def readTrainingData():
    training_data = pd.read_csv('./basic_model_data.csv')
    training_data['age_cat']=pd.qcut(training_data.first_arrest_Naissance,10,labels=False).astype('category')
    training_data['arrestedWithAdult']=(training_data['first_arrest_Adultes']>=2).astype(int)
    training_data['arrestedWithYouth']=np.sign(training_data['first_arrest_Jeunes'])
    training_data['arrestedWithSomeone']=np.sign(training_data['first_arrest_Adultes']+training_data['first_arrest_Jeunes'])
    training_data['first_arrest_Date']=pd.to_datetime(training_data.first_arrest_Date)
    training_data['first_arrest_year']=training_data.first_arrest_Date.dt.year
    # format data
    training_data.first_arrest_SeqE = training_data.first_arrest_SeqE.astype('category')
    training_data.first_arrest_SEXE = training_data.first_arrest_SEXE.astype('category')
    training_data.first_arrest_NCD1 = training_data.first_arrest_NCD1.astype('category')
    training_data.first_arrest_NCD2 = training_data.first_arrest_NCD2.astype('category')
    training_data.first_arrest_NCD3 = training_data.first_arrest_NCD3.astype('category')
    training_data.first_arrest_NCD4 = training_data.first_arrest_NCD4.astype('category')
    training_data.first_arrest_MUN = training_data.first_arrest_MUN.astype('category')
    training_data.first_arrest_ED1 = training_data.first_arrest_ED1.astype('category')
    training_data.second_arrest_SeqE = training_data.second_arrest_SeqE.astype('category')
    training_data.second_arrest_SEXE = training_data.second_arrest_SEXE.astype('category')
    training_data.second_arrest_NCD1 = training_data.second_arrest_NCD1.astype('category')
    training_data.second_arrest_NCD2 = training_data.second_arrest_NCD2.astype('category')
    training_data.second_arrest_NCD3 = training_data.second_arrest_NCD3.astype('category')
    training_data.second_arrest_NCD4 = training_data.second_arrest_NCD4.astype('category')
    training_data.second_arrest_MUN = training_data.second_arrest_MUN.astype('category')
    training_data.second_arrest_ED1 = training_data.second_arrest_ED1.astype('category')
    return training_data


   
#%%
