#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:42:20 2016

@author: andrewsonta

This script classifies 
"""


import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn
from sklearn import mixture
import os
import csv

print(__doc__)


#Import Clean Data
UIL_data_15min=pd.read_csv("UIL_data_15min",
                delim_whitespace=True,
                skipinitialspace=True)

# Set constants that determine the range of analysis
cleanData_15min = np.arange(0, len(UIL_data_15min))
cleanDataLen_15min = len(UIL_data_15min)
numDays = cleanDataLen_15min / (96)



# Define a function to take create n_steps x n_days matrices in python
def dayMatrix(mat):
    mat = np.resize(np.transpose(mat),(len(mat)/(96) ,(96)))
    return mat

# Load Data

UIL_data_15min_mat = UIL_data_15min.as_matrix(columns=UIL_data_15min.columns[:])
num_occ = len(UIL_data_15min_mat[0,:])

arraysDict = {}

for i in range(0,num_occ):
    arraysDict['x{0}'.format(i)] = dayMatrix(UIL_data_15min_mat[:,i])

#--------------------------------------------------------------------
# Cluster into 2 states intitially
# (inferred from component selection process)
#--------------------------------------------------------------------

# Iterates through every 96 timestep x 12 day array.

Y_full = np.ndarray(shape=(numDays*96, num_occ))
n_components_range = range(1, 3)

# Loop through each occupant
# in the following for loop, use either 1: , or, len(clean_dep_mat[0,:])): as
# the second arguments in the range() function

for i in range(0,len(UIL_data_15min_mat[0,:])):

    iter_array= np.transpose(arraysDict['x{0}'.format(i)])

    for j in range(0, len(iter_array[i,:])):

        lowest_bic = np.infty
        bic = []

        fitData = np.ndarray(shape = (96,1),
                             buffer = iter_array[:, j])

        # create gmm model of 1 and 2 components and pick the best one
        for n_components in n_components_range:
            gmm = mixture.GMM(n_components = n_components, covariance_type = 'full')
            gmm.fit(fitData)
            bic.append(gmm.bic(fitData))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        #print(best_gmm.means_)

        if best_gmm.n_components == 2:
            Y_ = best_gmm.predict(fitData)
            if best_gmm.means_[0] > best_gmm.means_[1]:
                Y_[Y_ == 0] = 2
                Y_[Y_ == 1] = 0
                Y_[Y_ == 2] = 1
        else:
            Y_ = np.zeros(len(fitData))

        Y_full[(96*j):(96*(j+1)), i] = np.transpose(Y_)

#--------------------------------------------------------------------
# Cluster high energy data into two states
# (inferred from secondary component selection process)
#--------------------------------------------------------------------

mode1_array = np.ndarray(shape=(7, 12))
mode2_array = np.ndarray(shape=(7, 12))

Y_full_round2 = copy.copy(Y_full)

for i in range(0,num_occ):
    for j in range(0,12):

        # find 96x1 vector within classified matrix
        data_to_classify = Y_full[(96*j):(96*(j+1)), i]

        # make copy preserving the index
        data_with_index = np.ndarray(shape = (96, 2))
        data_with_index[:,0] = range(96*j,96*(j+1))
        data_with_index[:,1] = data_to_classify

        # make copy without zeros
        data_with_index_nozeros = data_with_index[data_with_index[:,1] != 0]
        data_with_index_nozeros[:,0].astype(int)

        # only do the second round of clustering if there are more than two "on" data points
        if len(data_with_index_nozeros) > 2:

            # get the raw "on" data
            raw_on_data_extracted = UIL_data_15min_mat[data_with_index_nozeros[:,0].astype(int), i]

            # put it in the right format
            fitData2 = np.ndarray(shape = (len(raw_on_data_extracted),1),
                                 buffer = raw_on_data_extracted)

            # second gmm
            gmm2 = mixture.GMM(n_components = 2, covariance_type = 'full')
            gmm2.fit(fitData2)

            Y_2 = gmm2.predict(fitData2)

            # if the means are flipped, correct them
            if gmm2.means_[0] > gmm2.means_[1]:
                Y_2[Y_2 == 0] = 2
                mode1 = gmm2.means_[1]
                mode1_array[i,j] = gmm2.means_[1]
                mode2 = gmm2.means_[0]
                mode2_array [i,j] = gmm2.means_[0]

            else:
                Y_2[Y_2 == 1] = 2
                Y_2[Y_2 == 0] = 1
                mode1 = gmm2.means_[0]
                mode1_array[i,j] = gmm2.means_[0]
                mode2 = gmm2.means_[1]
                mode2_array [i,j] = gmm2.means_[1]

                # if there are outliers, reclassify them
            for k in range(0,len(fitData2-1)):

                if Y_2[k] == 1 and fitData2[k] > mode2:#np.mean(fitData2[Y_2==2]):
                    Y_2[k] = 2

                if Y_2[k] == 2 and fitData2[k] < mode1:#np.mean(fitData2[Y_2==1]):
                    Y_2[k] = 1


            # put the second round of clustering back into the dataset
            clustered_with_index_nozeros = data_with_index_nozeros
            clustered_with_index_nozeros[:,1] = Y_2

            Y_full_round2[clustered_with_index_nozeros[:,0].astype(int), i] = clustered_with_index_nozeros[:,1]


occupant = 7
day_1 = 4
day_2 = 5
daylight = 1

# Create a Transition Matrix
Transition = np.ndarray(shape=(95, 4))
time = 0 + 100*daylight
for k in range(0,95):

    if (k+1)/4 == round((k+1)/4):
        Transition[k,0] = time + 55
        time = time + 55
    else:
        Transition[k,0] = time + 15
        time = time + 15

    for i in range(1,96):
        if Y_full_round2[i+96*day_1,occupant-1] != Y_full_round2[i+96*day_1-1,occupant-1]:
            Transition[i-1, 1] = Y_full_round2[i+96*day_1,occupant-1]
            Transition[i-1, 2] = UIL_data_15min_mat[(i+96*day_1-1),occupant-1]
            Transition[i-1, 3] = UIL_data_15min_mat[i+96*day_1,occupant-1]
        else:
            Transition[i-1, 1] = 999
            Transition[i-1, 2] = UIL_data_15min_mat[(i+96*day_1-1),occupant-1]
            Transition[i-1, 3] = UIL_data_15min_mat[(i+96*day_1),occupant-1]

#%%
sd_array = np.ndarray(shape=(1, 7))

Occupant_1_on_data = UIL_data_15min_mat[Y_full[:,0] == 1,0]
sd_array[0,0] = np.std(Occupant_1_on_data)

Occupant_2_on_data = UIL_data_15min_mat[Y_full[:,1] == 1,1]
sd_array[0,1] = np.std(Occupant_2_on_data)

Occupant_3_on_data = UIL_data_15min_mat[Y_full[:,2] == 1,2]
sd_array[0,2] = np.std(Occupant_3_on_data)

Occupant_4_on_data = UIL_data_15min_mat[Y_full[:,3] == 1,3]
sd_array[0,3] = np.std(Occupant_4_on_data)

Occupant_5_on_data = UIL_data_15min_mat[Y_full[:,4] == 1,4]
sd_array[0,4] = np.std(Occupant_5_on_data)

Occupant_6_on_data = UIL_data_15min_mat[Y_full[:,5] == 1,5]
sd_array[0,5] = np.std(Occupant_6_on_data)

Occupant_7_on_data = UIL_data_15min_mat[Y_full[:,6] == 1,6]
sd_array[0,6] = np.std(Occupant_7_on_data)


with open('classified_data.csv', 'wb') as f:
   writer = csv.writer(f)
   writer.writerows(Y_full_round2)

