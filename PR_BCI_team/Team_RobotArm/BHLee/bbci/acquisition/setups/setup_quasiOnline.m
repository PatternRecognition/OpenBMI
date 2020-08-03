global EEG_TMP_DIR EEG_RAW_DIR VP_CODE TODAY_DIR LOG_DIR REMOTE_RAW_DIR
EEG_TMP_DIR = 'C:\data\eeg_temp\';
VP_CODE= '';

setup_bbci_bet_unstable; %% needed for acquire_bv
% First of all check that the parllelport is properly conneted.
bvr_checkparport('type','S');

fprintf('\n\nWelcome to the Quasi-Online study\n\n');
fprintf('BrainVision Recorder should be running otherwise start it and restart setup_quasiOnline.\n\n')

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));

%% Check whether a directory exists that is to be used
dd= dir([EEG_RAW_DIR 'VP??_' today_str '*']);
k= 0;
while isempty(VP_CODE) && k<length(dd),
  k= k+1;
  de= dir([EEG_RAW_DIR dd(k).name '\*.eeg']);
  if isempty(de),
    VP_CODE= dd(k).name(1:4);
    fprintf('!!Using existing directory <%s>!!\n', dd(k).name);
  end
end

%% Load Workspace into the BrainVision Recorder
error('fill in workspace name');
bvr_sendcommand('loadworkspace', 'XXX');
bvr_sendcommand('viewsignals');

%% Generate a Subject Code and folder name to save the EEG data in
letter_start= 'q';
letter_reserved= 'xyz';
while isempty(VP_CODE),
  dd= dir([EEG_RAW_DIR 'VP' letter_start '?_??_??_??*']);
%  letters_used= apply_cellwise({dd.name}, inline('x(4)','x'));
%  last_letter= char(max(letters_used));
  last_letter= dd(end).name(4);
  if last_letter~='z',
    VP_CODE= [VP letter_start last_letter+1];
  else
    letter_start= char(min(setdiff([char(letter_start+1):'z'], letter_reserved)));
  end
end

TODAY_DIR= [EEG_RAW_DIR VP_CODE '_' today_str '\'];
if ~exist(TODAY_DIR, 'dir'),
  mkdir_rec(TODAY_DIR);
end
LOG_DIR= [TODAY_DIR 'log\'];
if ~exist(LOG_DIR, 'dir'),
  mkdir_rec(LOG_DIR);
end
REMOTE_RAW_DIR= today_dir;

fprintf('EEG data will be saved in <%s>.\n', TODAY_DIR);

error('general_port_fields');

clear bbci
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'imag_lett*');
bbci.classDef= {1,      2;      ...
                'left', 'right'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.save_name= [TODAY_DIR 'classifier'];
bbci.player= 1;
