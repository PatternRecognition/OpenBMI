ppACQ_PREFIX_LETTER= '';
ACQ_LETTER_START= 'f';

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
  %VP_CODE = 'VPf';
end

path([BCI_DIR 'acquisition/setups'], path);

fprintf('\n\nWelcome to BBCI LRP experiment\n\n');
%system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
%bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
%bvr_sendcommand('loadworkspace', 'reducerbox_64std_noEye');

try
 bvr_checkparport('type','S');
catch
 error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
%TODAY_DIR= [EEG_RAW_DIR 'VPfbc_10_07_29\' ];
acq_makeDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;
LOG_DIR = [TODAY_DIR '\log\'];

%% prepare settings for classifier training
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.setup= 'LRP'; %'sellap';
bbci.clab= {'not','QQ*'};
bbci.classes= 'auto';
bbci.classDef= {1, 2, 3; 'left','right','foot'};
bbci.feedback= 'PYFF';
bbci.setup_opts.ilen_apply= 750;
bbci.adaptation.UC= 0.05;
bbci.adaptation.UC_mean= 0.11;
bbci.adaptation.UC_pcov= 0.001;
bbci.adaptation.load_tmp_classifier= 0;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;


cfy_name= 'classiClaudi.mat';

copyfile([EEG_RAW_DIR '/LRP Classifier/' cfy_name '*'],TODAY_DIR);


%VP_SCREEN= [-1023 0 1024 768];
%VP_SCREEN= [-799 0 800 600];
%VP_SCREEN= [-1279 0 1280 1024];
VP_SCREEN= [-1919 0 1920 1200];
%VP_SCREEN= [-1281 0 1280 1024];
%VP_SCREEN=[0 0 500 400];

fprintf('Now you can start by typing: run_LRPTest\n');
