if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/xtreme_eeg']);

fprintf('\n\nWelcome to Xreme EEG bike\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'Fahrrad_Berlin2.rwksp');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
mkdir([TODAY_DIR 'data']);

%VP_SCREEN= [-2560 0 1280 768];
%VP_SCREEN= [1280 0 1280 768];
VP_SCREEN= [-1920 0 1920 1200];

fprintf('Type ''run_xtreme_eeg'' and press <RET>\n');
