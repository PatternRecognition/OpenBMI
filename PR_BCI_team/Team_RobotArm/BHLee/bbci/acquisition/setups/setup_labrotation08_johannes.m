if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/labrotation08_johannes'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to Labrotation 08 (Johannes)\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'FastnEasy_auditoryP300');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_getDataFolder;

VP_SCREEN= [-1023 0 1024 768];
fprintf('Display resolution of secondary display must be set to 1024x768.\n');
fprintf('Type ''run_labrotation08_johannes'' and press <RET>.\n');
