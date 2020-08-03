if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end


path([BCI_DIR 'acquisition/setups/project09_ssvep'], path);
fprintf('\n\nWelcome to BBCI project09_ssvep\n\n');
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);

% Load Workspace into the BrainVision Recorder
%bvr_sendcommand('loadworkspace', ['season9_' lower(CLSTAG)]);
%bvr_sendcommand('loadworkspace', 'project09_ssvep_45ch');
bvr_sendcommand('loadworkspace', 'FastnEasy_occ_temp_dense_64ch');

bvr_sendcommand('stoprecording');
try
  bvr_checkparport('type','S');
catch
  warning('Check amplifiers (all switched on?) and trigger cables.');
end
global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder;
REMOTE_RAW_DIR= TODAY_DIR;

VP_SCREEN= [-1023 0 1024 768];
fprintf('Display resolution of secondary display must be set to 1024x768.\n');
fprintf('Type ''run_project09_ssvep'' and press <RET>.\n');