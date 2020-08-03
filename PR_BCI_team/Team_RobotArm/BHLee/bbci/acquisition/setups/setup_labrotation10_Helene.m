if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

VP_SCREEN = [0 0 1280 1024]
addpath([BCI_DIR 'acquisition/setups/labrotation10_Helene']);
addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\nWelcome to labrotation10_Helene (Symmetry Experiment)\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_linked_mastoids');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
mkdir([TODAY_DIR 'data']);

fprintf('Type ''run_labrotation10_Helene'' and press <RET>\n');

