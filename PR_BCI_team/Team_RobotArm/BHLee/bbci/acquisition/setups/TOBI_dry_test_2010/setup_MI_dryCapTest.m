

global CLSTAG
CLSTAG= 'FR';
if isempty(CLSTAG),
  error('Variable CLSTAG has to be defined');
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/season9'], path);

fprintf('\n\nWelcome to BBCI Season 9\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);

% Load Workspace into the BrainVision Recorder
%bvr_sendcommand('loadworkspace', ['season9_' lower(CLSTAG)]);
% bvr_sendcommand('loadworkspace', 'season9');

bvr_sendcommand('loadworkspace', 'headbox_nouzz_19');
bvr_sendcommand('stoprecording');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder;
REMOTE_RAW_DIR= TODAY_DIR;

%% prepare settings for classifier training
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
all_classes= {'left', 'right', 'foot'};
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
bbci= [];
bbci.setup= 'cspauto';
% bbci.clab= {'not','E*','Fp*','AF*','PO*','O*'};
bbci.classes= all_classes([ci1 ci2]);
bbci.classDef= cat(1, {1, 2}, bbci.classes);
bbci.feedback= '1d';
bbci.train_file= strcat(subdir, '/imag_fbarrow*');
bbci.setup_opts.ilen_apply= 500;
bbci.adaptation.UC= 0.1;
bbci.adaptation.UC_mean= 0.075;
bbci.adaptation.UC_pcov= 0.001;
bbci.adaptation.delay= 1000;
bbci.adaptation.load_tmp_classifier= 1;
%% make bbci_default available in the run_script
global bbci_default
bbci_default= bbci;

%cfy_name= ['/bbci_classifier_sellap_' VP_CODE '_setup_001'];
%copyfile([EEG_RAW_DIR '/pretrained_classifiers/season8/' cfy_name '*'], ...
%  TODAY_DIR);

VP_SCREEN= [-1023 0 1024 768];
fprintf('Display resolution of secondary display must be set to 1024x768.\n');
fprintf('Type ''run_season9'' and press <RET>.\n');
