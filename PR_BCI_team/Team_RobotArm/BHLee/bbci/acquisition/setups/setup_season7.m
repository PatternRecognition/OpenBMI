path([BCI_DIR 'acquisition/setups/season7'], path);

global EEG_TMP_DIR EEG_RAW_DIR VP_CODE TODAY_DIR
EEG_TMP_DIR = 'C:\data\eeg_temp\';
VP_CODE= '';

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to Season 7\n\n');
fprintf('BrainVision Recorder should be running otherwise start it and restart setup_season7.\n\n')

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'season7');
bvr_sendcommand('viewsignals');
% Check that the parllelport is properly conneted.
bvr_checkparport('type','S');

acq_getDataFolder;

fprintf('EEG data will be saved in <%s>.\n', TODAY_DIR);
