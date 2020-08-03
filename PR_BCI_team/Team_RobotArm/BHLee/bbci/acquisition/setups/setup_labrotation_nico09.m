if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

addpath([BCI_DIR 'acquisition/setups/labrotation_nico09']);
addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\n\nWelcome to covert attention\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_EOGvu.rwksp');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
mkdir([TODAY_DIR 'data']);

VP_SCREEN= [-1920 0 1920 1200];

fprintf('Type ''run_labrotation_nico09'' and press <RET>\n');
