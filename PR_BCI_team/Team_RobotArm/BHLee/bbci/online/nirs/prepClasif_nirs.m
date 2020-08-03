%Do startup, initialization, and creation of a bbci compatible classifier:

cd('C:\Tanis');
%Do the starup
startup_nirs

%Load subject information:
%Valid ones for EEG:
%VPeae_10_03_05
VPeaa_10_01_13_nirs

%NIRS



%Prepare classifier - load raw data into Cnt, mrk, mnt, also add to bbci
bbci_bet_prepare_nirs

%Analyze data with classifier - artifacts, selection of temporal filter, time interval, spatial filter, xvalidation of features
bbci_bet_analyze

%Compute clasifier - calculates the classifier
bbci_bet_finish

%Compute bbci_apply for offline situation
%(the file created by running the subject is given as setup_list thtough the varargin field)
bbci_bet_apply_offline_nirs(Cnt,mrk_orig,'setup_list',[save_file '.mat'])