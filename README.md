# GLMCC
GLMCC: The generalized linear model for spike cross-correlation 

## Outline
This program estimates interneuronal connections by fitting a generalized linear model (GLM) to spike cross-correlations (Kobayashi et al., Nature Communications 2019). A ready-to-use version of the web application is available at [our website](https://s-shinomoto.com/CONNECT/).
For details, see [here](https://www.nature.com/articles/s41467-019-12225-2).

## Requirement
Python3
Numpy 1.10.4
Scipy 0.17.0
Matplotlib 1.5.1

You can install Numpy, Scipy and Matplotlib by using the following command.

$ pip install numpy

$ pip install scipy

$ pip install matplotlib

Multi-platform support

   * Windows
   * Mac OS X

## Getting started
You can use the following command to clone from Git's repository.

$ git clone 'unknown'


## Experimental data
Spike data is in the two directories: 

* simulation_data
* experimental_data


## Usage

glmcc.py:

glmcc.py has all the necessary functions to estimate interneuronal connections by fitting GLM to spike cross-correlations. 
You can run this code to plot the spike cross-correlation: 

$ python3 glmcc.py CC <f_ref> <f_tar> T(s)

where, <f_ref> (<f_tar>) is the file of spike times of the reference (target) neuron, and T (s) is the duration of recording. It is assumed that <f_ref> and <f_tar> are in the same directry as "glmcc.py". This code generates an image that shows the histogram of Cross correlogram. 

You can also run this code to fit the GLM to spike cross-correlations: 

$ python3 glmcc.py (GLM or LR) <f_ref> <f_tar> tau_+ tau_- gamma T(s)

where, tau_+ (tau_-) is the time constant after (before) the spike times of the reference neuron,  gamma (the defualt parameter was gamma= 0.0005) is the hyper-parameter controling the faltness of a(t) (See Eq.(8) in the paper). This code generates an image file (GLMCC_<f_ref>_<f_tar>.png) that compares the histogram of Cross correlogram to its fit by GLMCC. If you choose GLM, original GLMCC will run. If you choose LR, revised version of GLMCC (see CoNNECT: Convolutional Neural Network for Estimating synaptic Connectivity from spike Trains) will run. For example, you can fit the cross-correlation from "cell9" and "cell4": 

$ python3 glmcc.py GLM  cell9.txt  cell4.txt  4  4  0.0005  5400


Est_Data.py:

This program estimates the connectivity matrix among the neurons in a directory. 
You can run: 

$ python3 Est_Data.py <Directory of the data> <the number of neurons> (sim or exp) (GLM or LR)

where, sim (exp) corresponds to simulated (experimental) data, and GLM (LR) corresponds to original GLMCC (revised GLMCC). The output file "W_py_5400.csv" is the estimate of the connectivity matrix (in the units of the post-synaptic potential). The column (row) represents the index of the pre(post)-synaptic neuron. 
For example, you can analyze the simulation data in "simulation_data":  

$ python3 Est_Data.py simulation_data 20 sim GLMCC

glmcc_fitting.py: 
This program generates a Cross-correlation figure for each pair of neurons. 
After you run "Est_Data.py", you can run: 

$ python3 glmcc_fitting.py <the number of neurons> <Directory of the data> (sim or exp)  <Wfile>  all (GLM or LR)

where, sim (exp) corresponds to simulated (experimental) data, GLM (LR) corresponds to original GLMCC (revised GLMCC), and <Wfile> is the file name of the estimated connectivity matrix (e.g., W_py_5400.csv). Please note that this code requires an estimate of the connectivity matrix and you have to wait some time for plotting the cross-correlation. It takes around 5 mins when we analyze simulated data (20 neurons). The figure file will be "allCC.png". The column (row) represents the index of the post (pre)-synaptic neuron (transposed matrix of "W_py_5400.csv"). 
For example, you can analyze the simulation data in "simulation_data": 

$ python3 glmcc_fitting.py 20 simulation_data sim  W_py_5400.csv  all GLMCC


## Licence
MIT

Please contact [Ryota Kobayashi](http://www.hk.k.u-tokyo.ac.jp/r-koba/en/contact.html) if you want to use the code for commercial purposes.


## The program was developed by
Junichi Haruna and Masahiro Naito
