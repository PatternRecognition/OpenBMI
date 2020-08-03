if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/projekt_biomed09']);
addpath([BCI_DIR 'acquisition/setups/season10']);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n*********************\nWelcome to biomed09 \n*********************\n');

%% Start Brainvision recorder
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_VisualSetup_EOGvu');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Make folder for saving EEG data
acq_makeDataFolder;
global TODAY_DIR

fprintf('Type ''run_biomed09'' and press <RET>\n');
