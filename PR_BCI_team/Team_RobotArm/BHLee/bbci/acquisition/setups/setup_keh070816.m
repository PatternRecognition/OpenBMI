path(path, [BCI_DIR 'acquisition/setups/lmu_2008_04_08']);

global EEG_TMP_DIR EEG_RAW_DIR VP_CODE TODAY_DIR LOG_DIR REMOTE_RAW_DIR
EEG_TMP_DIR = 'C:\data\eeg_temp\';

setup_bbci_bet_unstable; %% needed for acquire_bv
% First of all check that the parllelport is properly conneted.
%bvr_checkparport('type','S');
%
fprintf('\n\nWelcome to the KEH Experiment on 16th Aug 2007\n\n');
fprintf('BrainVision Recorder should be running otherwise start it and restart setup_quasiOnline.\n\n')

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));

VP_CODE= 'VPhc';
TODAY_DIR= [EEG_RAW_DIR VP_CODE '_' today_str '\'];

if ~exist(TODAY_DIR, 'dir'),
  mkdir_rec(TODAY_DIR);
end
LOG_DIR= [TODAY_DIR 'log\'];
if ~exist(LOG_DIR, 'dir'),
  mkdir_rec(LOG_DIR);
end
REMOTE_RAW_DIR= TODAY_DIR;

fprintf('EEG data will be saved in <%s>.\n', TODAY_DIR);

%error('define general_port_fields');

clear bbci
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'real_lett_mundVPhc');
bbci.classDef= {1,      2;      ...
                'left', 'right'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.save_name= [TODAY_DIR 'classifier'];
bbci.player= 1;
