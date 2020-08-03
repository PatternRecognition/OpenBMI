%global CLSTAG
%if isempty(CLSTAG),
%  error('Variable CLSTAG has to be defined');
%end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/season10'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to BBCI\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
%bvr_sendcommand('loadworkspace', ['season10_64ch_FastnEasy_EMG' lower(CLSTAG) '_EOG']);
bvr_sendcommand('loadworkspace', ['reducerbox_64mcc_noEOG.rwksp']);
pause(2);

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];

%% prepare settings for classifier training
%all_classes= {'left', 'right', 'foot'};
%ci1= find(CLSTAG(1)=='LRF');
%ci2= find(CLSTAG(2)=='LRF');
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.setup= 'sellap';
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','PO*','O*'};
%bbci.classes= all_classes([ci1 ci2]);
%ATTENTION, THE CLASSES ARE FIXED TO LR
bbci.classes= 'LR';
bbci.classDef= {1, 2, 3; 'left','right','foot'};
bbci.feedback= '1d';
bbci.setup_opts.ilen_apply= 750;
bbci.adaptation.UC= 0.05;
bbci.adaptation.UC_mean= 0.075;
bbci.adaptation.UC_pcov= 0.03;
bbci.adaptation.load_tmp_classifier= 1;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;

cfy_name= ['Lap_C3z4_bp2_' bbci.classes '.mat'];
tmpcls=load([EEG_RAW_DIR '/subject_independent_classifiers/season10/' cfy_name]);
tmpcls.cls.mrk_start={1,2};
save([TODAY_DIR cfy_name],'-struct','tmpcls');
%copyfile([EEG_RAW_DIR '/subject_independent_classifiers/season10/' cfy_name], ...
%  TODAY_DIR);

%VP_SCREEN= [-799 0 800 600];
VP_SCREEN= [-1279 900-1024+1 1280 1024-19];
