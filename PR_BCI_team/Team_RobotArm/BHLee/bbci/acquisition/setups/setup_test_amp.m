global EEG_TMP_DIR EEG_RAW_DIR TODAY_DIR
EEG_TMP_DIR = 'C:\data\eeg_temp\';


setup_bbci_bet_unstable; %% needed for acquire_bv
% First of all check that the parllelport is properly conneted.
bvr_checkparport('type','S');

fprintf('\n\nWelcome to Season 7\n\n');
fprintf('BrainVision Recorder should be running otherwise start it and restart setup_season7.\n\n')

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));


%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'season7');
bvr_sendcommand('viewsignals');

    
TODAY_DIR= [EEG_RAW_DIR 'AMP_Test'  '_' today_str '\'];
if ~exist(TODAY_DIR, 'dir'),
  mkdir_rec(TODAY_DIR);
end

fprintf('EEG data will be saved in <%s>.\n', TODAY_DIR);
