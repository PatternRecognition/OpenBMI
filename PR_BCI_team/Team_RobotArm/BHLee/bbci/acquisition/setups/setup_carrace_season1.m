if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/carrace_season1']);
addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\n\nWelcome to BFNT-A1 & brain@work - Car Race Season 1\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'reducerbox_64std_EMGf.rwksp');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
mkdir([TODAY_DIR 'data']);

%VP_SCREEN= [-2560 0 1280 768];
VP_SCREEN= [1280 0 1280 768];
fprintf('Type ''run_carrace_season1'' and press <RET>\n');
