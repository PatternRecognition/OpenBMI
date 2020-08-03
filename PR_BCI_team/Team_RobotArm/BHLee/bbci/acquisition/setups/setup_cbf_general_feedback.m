%VP_CODE= 'VPdb';

global CLSTAG
CLSTAG= 'LR';
if isempty(CLSTAG),
  error('Variable CLSTAG has to be defined');
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/cbf_general_feedback'], path);

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
error('fill in the name of zour workspace in the following line.')
bvr_sendcommand('loadworkspace', 'YOUR_WORKSPACE');

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
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.setup= 'sellap';
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','PO*','O*'};
bbci.classDef= {1, 2, 3; 'left','right','foot'};
bbci.classes= bbci.classDef(2,[ci1 ci2]);
bbci.feedback= '1d';
bbci.setup_opts.ilen_apply= 750;
bbci.adaptation.UC= 0.05;
bbci.adaptation.UC_mean= 0.11;
bbci.adaptation.UC_pcov= 0.001;
bbci.adaptation.load_tmp_classifier= 0;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;

cfy_name= 'Lap_C3z4_bp2';
copyfile([EEG_RAW_DIR '/subject_independent_classifiers/season10/' cfy_name '*'], ...
  TODAY_DIR);

VP_SCREEN= [-1279 0 1280 1024];
%VP_SCREEN= [-1919 0 1920 1200];

fprintf('Now you can start: run_cbf_general_feedback\n');
