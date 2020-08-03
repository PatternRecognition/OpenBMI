%This directory contains some scripts that demonstrate the correct use
%of functions in the neuro_toolbox. Many demos use known data sets
%which are, e.g., stored in a subdirectory of
%  /home/schlauch/blanker/Daten/
%e.g., uci or ida_training. You are welcome to store more benchmark
%data sets there, e.g, from NIPS competitions. Please supply also
%functions to load the data into the usual fv struct. For the uci
%data such a function is load_uci_dataset. To load data from ascii
%files there exists a function called load_features_from_ascii.
%To make the demos work the global variable DATA_DIR must be set
%to '/home/schlauch/blanker/Daten/' or another directory containing
%the subdirectories that are used by the demos. The script
%startup_idabox first looks whether it finds an environment variable
%in the shell called 'DATA'. If so it uses that as DATA_DIR. Otherwise
%it checks whether '/home/schlauch/blanker/Daten/' is available to
%use that as DATA_DIR. If both attempts fail the variable DATA_DIR
%is empty. In this case, either you have to define the environment
%variable $DATA appropriately before starting matlab, e.g., in your
%~/.bashrc, or you have to define the global variable DATA_DIR
%by yourself before starting the demos.
%
% demo_feature_selection - This demo uses a very simple method to
%   select features, the Fisher criterion. The point of the demo is
%   to demonstrate how to validation a feature selection method.
%   This demo also shows how to handle free variables in the
%   feature pre-processing.
%
% demo_outlier_removal - This demo uses a very simple method to
%   remove outliers, the malahanobis distance to the class centers. 
%   The point of the demo is to demonstrate how to validation an 
%   outlier removal method. This demo also shows how to handle free
%   variables in the feature pre-processing.
%
% demo_proc_trainApply - This demo shows how to define different
%   feature processings for training and test set.
