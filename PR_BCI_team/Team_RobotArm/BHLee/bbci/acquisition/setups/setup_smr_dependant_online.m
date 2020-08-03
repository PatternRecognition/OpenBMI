if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/smr_neurofeedback_season2'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to SMR Neurofeedback Season2\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'smr_dependant_online');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder('log_dir',1, 'multiple_folders', 1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];

%% prepare settings for classifier training
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.setup= 'cspauto';
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','PO*','O*'};
bbci.classes= {'left','right'};
bbci.classDef= {1, 2; 'left','right'};
bbci.feedback= '';
bbci.setup_opts.ilen_apply= 750;
bbci.adaptation.running= 0;
bbci.adaptation.UC= 0.05;
bbci.adaptation.UC_mean= 0.075;
bbci.adaptation.UC_pcov= 0.03;
bbci.adaptation.load_tmp_classifier= 1;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;

cfy_name= 'Lap_C3z4_bp2';
copyfile([EEG_RAW_DIR '/subject_independent_classifiers/season10/' cfy_name '*'], ...
  TODAY_DIR);

VP_SCREEN= [1920 0 1920 1200];
fprintf('Type ''run_smr_dependant_online'' and press <RET>.\n');
