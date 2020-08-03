%cd([BCI_DIR 'acquisition/setups/special_event']);

VP_CODE= 'VPzk';
CLSTAG= 'LR';

path([BCI_DIR 'acquisition/setups/special_event'], path);
path([BCI_DIR 'acquisition/setups/vitalbci_season1'], path);

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', ['reducerbox_64mcc_noEOG']);

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
bbci.setup_opts.ilen_apply= 500;
bbci.adaptation.UC= 0.05;
bbci.adaptation.UC_mean= 0.11;
bbci.adaptation.UC_pcov= 0.001;
bbci.adaptation.load_tmp_classifier= 1;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;

cfy_name= 'Lap_C3z4_bp2';
copyfile([EEG_RAW_DIR '/subject_independent_classifiers/season10/' cfy_name '*'], ...
  TODAY_DIR);

fprintf('Now you can start: run_flipper_2009_11_05\n');

%VP_SCREEN= [-799 0 800 600];
VP_SCREEN= [-1919 0 1920 1200];
