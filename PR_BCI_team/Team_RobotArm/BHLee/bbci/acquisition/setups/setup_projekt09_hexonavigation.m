if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/project09_hexonavigation'], path);

fprintf('\n\nWelcome to the Project Hex-o-Navigation.\n\n');

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

VP_SCREEN= [-1279 900-1024+1 1280 1024-19];
fprintf('\nDisplay resolution of external display must be set to 1280x1024.\n');
fprintf('Type ''run_projekt09_hexonavigation'' and press <RET>.\n\n');
fprintf('If you want to execute only some trials, maybe because the experiment\n');
fprintf('broke down before, set e.g. which_trials = 5:7\n');
fprintf('trials that don''t exist are ignored, so it is safe to type\n');
fprintf('for example which_trials = 8:999\n\n');
