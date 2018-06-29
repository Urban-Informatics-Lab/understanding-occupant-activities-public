# understanding-occupant-activities-public
This repository contains code for our paper: [Understanding building occupant activities at scale: An integrated knowledge-based and data-driven approach. Advanced Engineering Informatics 37, 1-13.](https://doi.org/10.1016/j.aei.2018.04.009)

## primary-component-selection.py
This file performs the primary component selection process as described in the paper. It reads a time-series dataset (specifications below), and it fits varitional Bayesian Gaussian mixture models (up to 5 components each) to each sub-dataset (defined as time series for one occupant for one day). It prints a histogram showing how many components were chosen for each sub-dataset.

## secondary-component-selection.py
If the primary componenet selection process results in 2 components, then this file should be run (as described in the paper). It completes a similar process as in ``primary-component-selection.py``, but only for the higher-energy data as classified through a two-component Gaussian mixture model.

## classification.py
This file completes the classification process, using a Gaussian mixture model. It is currently set up to first fit a 2-component Gaussian mixture model to each sub-dataset (for each occupant for each day), separate the higher energy data, and then fit another 2-component Gaussian mixture model to the higher-energy sub-datasets. This two-step, two-component set up is based on the results from the component selection processes. See the paper for details.

## Data
The files are currently set up to read the ``UIL_data_15min`` file, which contains 12x96 rows and 7 columns, where the columns indicate the 7 occupants and the rows indicate the time steps. The data and the code are set up for 15-minute energy data, where 1 day contains 96 data points for each occupant.
