%% Basic setup
% The graphics output of the laptop is connected to a splitter box driving two monitors.
% These monitors extend the Desktop to the left, meaning that x-coordinates specifying 
% positions on that monitors will be negative. Specify the left border:
VP_SCREEN= [-2048 0 1024 800];
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

% Odd-numbered subjects (ie, VP 1,3,5..) should begin
% without noise(0), even numbered subjects (2,4,6...) should begin with
% noise(1)
% global noise
% noise = input('Begin Leitstand experiment with(1) or without(0) noise?')

addpath([BCI_DIR 'acquisition/setups/leitstand09']);
addpath([BCI_DIR 'acquisition/setups/season10']);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n*********************\nWelcome to brain@work\nLeitstand09 \n*********************\n');

%% Start Brainvision recorder
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_Leitstand09');
%bvr_sendcommand('loadworkspace', 'V-Amp_visual_P300'); % for testing

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Make folder for saving EEG data
acq_makeDataFolder;
global TODAY_DIR
%mkdir([TODAY_DIR 'data']);

fprintf('Type ''run_leitstand09'' and press <RET>\n');