#
'''
secondary-component-selection.py: This program performs the secondary component selection process.

Copyright (C) 2017-2018 Andrew J. Sonta, Perry E. Simmons, Rishee K. Jain     

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU Affero General Public License as published by the Free Software Foundation, either version 
3 of the License, or (at your option) any later version.      

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU Affero General Public License for more details. You should have received a copy of 
the GNU Affero General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import numpy as np
import pandas as pd
import copy
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn
from sklearn import mixture
import os

print(__doc__)


#%% Import Clean Data
# Requires file "UIL_data_15min" in the same directory
UIL_data_15min=pd.read_csv("UIL_data_15min",
                delim_whitespace=True,
                skipinitialspace=True)

numOcc = 7

# Set constants that determine the range of analysis
cleanData = np.arange(0, len(UIL_data_15min))
cleanDataLen = len(UIL_data_15min)
numDays = cleanDataLen / 96

#%% Function Definition

# Define a function to take create n_steps x n_days matrices in python
def dayMatrix(mat):
    mat = np.resize(np.transpose(mat),(len(mat)/96 ,96))
    return mat

#%% Load Data

UIL_data_15min_mat = UIL_data_15min.as_matrix(columns=UIL_data_15min.columns[1:])

arraysDict = {}

for i in range(0,len(UIL_data_15min_mat[1,:])):
    arraysDict['x{0}'.format(i)] = dayMatrix(UIL_data_15min_mat[:,i])

occ_1 = np.transpose(arraysDict['x0'])
occ_1[:,0]

#%% Cluster into initial two states

# Iterates through every 96 timestep x 41 day array.
# Can use to calculate aspects of feature vector

Y_full = np.ndarray(shape=(len(UIL_data_15min), numOcc))
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
            gmm = mixture.GaussianMixture(n_components = n_components, covariance_type = 'full')
            gmm.fit(fitData)
            bic.append(gmm.bic(fitData))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        # print(best_gmm.means_)

        if best_gmm.n_components == 2:
            Y_ = best_gmm.predict(fitData)
            if best_gmm.means_[0] > best_gmm.means_[1]:
                Y_[Y_ == 0] = 2
                Y_[Y_ == 1] = 0
                Y_[Y_ == 2] = 1
        else:
            Y_ = np.zeros(len(fitData))

        Y_full[(96*j):(96*(j+1)), i] = np.transpose(Y_)

#---------------------------------------------------------------------
# Secondary Component Selection Process
#---------------------------------------------------------------------

# Iterates through every 96 timestep x 12 day array.
# Requires customization to increase or decrease the number of possible
# components used by the variational Bayesian GMM

# Prints a histogram of components chosen

Y_full_round2 = copy.copy(Y_full)

all_k = np.ndarray(shape=(numDays, numOcc))
percent_ones = []
percent_twos = []
percent_threes = []

running_total = 0

for i in range(0,len(UIL_data_15min_mat[0,:])):

    k = []

    for j in range(0, len(iter_array[i,:])):

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
        if len(data_with_index_nozeros) > 5:

            # get the raw "on" data
            raw_on_data_extracted = UIL_data_15min_mat[data_with_index_nozeros[:,0].astype(int), i]

            # put it in the right format
            fitData2 = np.ndarray(shape = (len(raw_on_data_extracted),1),
                                 buffer = raw_on_data_extracted)

            # second gmm
            gmm2 = mixture.BayesianGaussianMixture(n_components = 5, covariance_type = 'full')
            gmm2.fit(fitData2)

            Y_2 = gmm2.predict(fitData2)

            # print gmm2.weights_
            # print gmm2.means_
            # print gmm2.covariances_

            # ax = plt.figure()
            x = np.array([np.linspace(np.min(fitData2), np.max(fitData2), 500)])
            x = np.transpose(x)

            num_means = 0

            for m, (mean, covar) in enumerate(zip(gmm2.means_, gmm2.covariances_)):
                if not np.any(gmm2.predict(x) == m):
                    continue
                num_means += 1
                # plt.plot(x, scipy.stats.norm.pdf(x, mean, np.sqrt(covar)))

            k.append(num_means)


            # plt.hist(fitData2, 50, normed=True, color='Blue')
            # plt.show()


            # # if the means are flipped, correct them
            # if gmm2.means_[0] > gmm2.means_[1]:
            #     Y_2[Y_2 == 0] = 2
            #
            # else:
            #     Y_2[Y_2 == 1] = 2
            #     Y_2[Y_2 == 0] = 1

            # put the second round of clustering back into the dataset
            clustered_with_index_nozeros = data_with_index_nozeros
            clustered_with_index_nozeros[:,1] = Y_2

            Y_full_round2[clustered_with_index_nozeros[:,0].astype(int), i] = clustered_with_index_nozeros[:,1]

            running_total += 1.0
            # print running_total

        else:
            k.append(0)



    all_k[:, i] = k
    percent_ones.append(k.count(1))
    percent_twos.append(k.count(2))
    percent_threes.append(k.count(3))


ones = np.sum(percent_ones) / (running_total)
twos = np.sum(percent_twos) / (running_total)
threes = np.sum(percent_threes) / (running_total)

print twos

plt.bar([1, 2, 3, 4, 5], [ones, twos, threes, 0, 0])
plt.show()
