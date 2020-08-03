function [cnt, mrk, sbjData] = load_calibration_data_Vital_BCI(VPDir, varargin)
global EEG_MAT_DIR
% create the path to calibration data
sbj = strtok(VPDir, '_');
file_name = ['imag_arrow' sbj '.mat'];
% load calibration data
[cnt, mrk] = eegfile_loadMatlab(fullfile(VPDir, file_name));

% load the classifier (take the last if there are multiple)
classi_files = dir(fullfile(EEG_MAT_DIR, VPDir, ['bbci_classifier_setup_*.mat']));
try
    S = load(fullfile(EEG_MAT_DIR, VPDir, classi_files(end).name));
catch
    error(['ERROR (Subject: ' sbj '), Unable to read file ' VPDir 'bbci_classifier_setup_*.mat'])
end
% get csp paramters that were used during feedback
sbjData = [];
sbjData.ival = S.bbci.setup_opts.ival;
sbjData.band = S.bbci.setup_opts.band;
sbjData.nPat = S.bbci.setup_opts.nPat;
sbjData.clab = S.bbci.clab;
sbjData.classes = S.bbci.classes;
sbjData.bbci = S.bbci;
    