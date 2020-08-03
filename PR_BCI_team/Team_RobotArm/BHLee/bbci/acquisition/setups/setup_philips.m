if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/philips']);
addpath([BCI_DIR 'acquisition/setups/season10']);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n*********************\nWelcome to Philips experiment \n*********************\n');
VP_SCREEN = [0 0 1440 900];
%% Start Brainvision recorder
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Make folder for saving EEG data
% Generate VP CODE (if necessary), and make acquisition folder. Multiple
% folders per day allowed.
acq_makeDataFolder();
% acq_makeDataFolder('multiple_folders',1);
global TODAY_DIR

fprintf('Type ''run_philips'' and press <RET>\n');
