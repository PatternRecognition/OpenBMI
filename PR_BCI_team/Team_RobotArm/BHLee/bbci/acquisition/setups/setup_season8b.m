global CLSTAG
if isempty(CLSTAG),
  error('Variable CLSTAG has to be defined');
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/season8'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to BBCI Season 8\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', ['FastnEasy_motor_EMG' lower(CLSTAG) '_EOG']);

 try
   bvr_checkparport('type','S');
 catch
   error('Check amplifiers (all switched on?) and trigger cables.');
 end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

%% prepare settings for classifier training
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
all_classes= {'left', 'right', 'foot'};
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
bbci= [];
bbci.setup= 'sellap';
bbci.clab= {'not','E*','Fp*','AF*','PO*','O*'};
bbci.classes= all_classes([ci1 ci2]);
bbci.classDef= cat(1, {1, 2}, bbci.classes);
bbci.feedback= '1d';
bbci.setup_opts.ilen_apply= 500;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;

cfy_name= ['/bbci_classifier_sellap_' VP_CODE];
copyfile([EEG_RAW_DIR '/pretrained_classifiers/season8b/' cfy_name '*'], ...
  TODAY_DIR);

VP_SCREEN= [-1023 0 1024 768];
fprintf('Display resolution of secondary display must be set to 1024x768.\n');
fprintf('Type ''run_season8'' and press <RET>.\n');
