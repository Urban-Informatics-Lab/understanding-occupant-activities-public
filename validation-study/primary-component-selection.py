#
'''
primary-component-selection.py: This program performs the primary component selection process.

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

# Set working directory for python
os.getcwd()

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

# Define a function to take create n_steps x n_days matrices in python
def dayMatrix(mat):
    mat = np.resize(np.transpose(mat),(len(mat)/96 ,96))
    return mat

# Load Data

UIL_data_15min_mat = UIL_data_15min.as_matrix(columns=UIL_data_15min.columns[1:])
arraysDict = {}

for i in range(0,len(UIL_data_15min_mat[1,:])):
    arraysDict['x{0}'.format(i)] = dayMatrix(UIL_data_15min_mat[:,i])

occ_1 = np.transpose(arraysDict['x0'])
occ_1[:,0]

#---------------------------------------------------------------
# Component Selection
#---------------------------------------------------------------

# Iterates through every 96 timestep x 12 day array.
# Requires customization to increase or decrease the number of possible
# components used by the variational Bayesian GMM

# Prints a histogram of components chosen

Y_full = np.ndarray(shape=(len(UIL_data_15min), numOcc))

# Loop through each occupant
# in the following for loop, use either 1): , or, len(clean_dep_mat[0,:])): as
# the second arguments in the range() function

all_k = np.ndarray(shape=(len(UIL_data_15min)/96, numOcc))
percent_ones = []
percent_twos = []
percent_threes = []

for i in range(0,len(UIL_data_15min_mat[0,:])):

    k = []

    iter_array= np.transpose(arraysDict['x{0}'.format(i)])

    # for each day
    for j in range(0, len(iter_array[i,:])):

        lowest_bic = np.infty
        bic = []

        fitData = np.ndarray(shape = (96,1),
                             buffer = iter_array[:, j])



        # create gmm model of 1 and 2 components and pick the best one
        gmm = mixture.BayesianGaussianMixture(n_components = 6,
                                          covariance_type = 'full',
                                          weight_concentration_prior = 1)

        gmm.fit(fitData)


        # ax = plt.figure()
        x = np.array([np.linspace(np.min(fitData), np.max(fitData), 500)])
        x = np.transpose(x)

        num_means = 0

        for m, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
            if not np.any(gmm.predict(x) == m):
                continue
            # plt.plot(x, scipy.stats.norm.pdf(x, mean, np.sqrt(covar)))
            num_means += 1

        k.append(num_means)

        # plt.hist(fitData, 50, normed=True, color='Blue')
        # plt.show()

    all_k[:, i] = k
    percent_ones.append(k.count(1))
    percent_twos.append(k.count(2))
    percent_threes.append(k.count(3))




    # print percent_twos

print all_k

# fig, ax = plt.subplots()
# index = np.arange(48)
# bar_width = 0.3
# opacity = 0.6
#
# rects1 = plt.bar(index, [x / 41.0 for x in percent_ones], bar_width, alpha = opacity, color = 'b', label = 'One Component')
# rects2 = plt.bar(index + bar_width, [x / 41.0 for x in percent_twos], bar_width, alpha = opacity, color = 'r', label = 'Two Components')
# rects3 = plt.bar(index + bar_width*2, [x / 41.0 for x in percent_threes], bar_width, alpha = opacity, color = 'g', label = 'Three Components')
# plt.legend()
# plt.xticks(index + bar_width / 2, [str(x) for x in index])
# plt.show()

ones = np.sum(percent_ones) / float(numDays*numOcc)
twos = np.sum(percent_twos) / float(numDays*numOcc)
threes = np.sum(percent_threes) / float(numDays*numOcc)

print ones, twos, threes

plt.bar([1, 2, 3, 4, 5], [ones, twos, threes, 0, 0])
plt.show()



# # PLOT THE GAUSSIAN MIXTURE MODEL AND DATA
# pdf = np.exp(logprob)
# pdf_individual = responsibilities * pdf[:, np.newaxis]
# ax.hist(fitData, 100, normed=True, histtype='stepfilled', alpha = 0.4)
# ax.plot(x, pdf, '-k')
# ax.plot(x, pdf_individual, '--k')
# ax.text(0.04, 0.96, "Best-fit Mixture", ha='left', va='top', transform = ax.transAxes)
# ax.set_xlabel('$t$')
# ax.set_ylabel('$p(x)$')
# plt.show()

# HISTOGRAM OF THE NUMBER OF COMPONENTS
# xaxis = np.array([1, 2, 3])
# plt.hist(k, 3, normed=1)
# plt.xlabel('Number of Components')
# plt.xticks(xaxis + 0.5, ['1','2','3'])
# plt.ylabel('Frequency')
# plt.show()
