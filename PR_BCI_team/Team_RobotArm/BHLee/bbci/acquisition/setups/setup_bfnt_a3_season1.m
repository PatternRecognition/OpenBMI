if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/bfnt_a3_season1'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to BFNT-A3 - Season 1\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'FastnEasy_muscog_season2');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;

%VP_SCREEN= get(0,'ScreenSize');
VP_SCREEN= [-1919 0 1920 1200];
fprintf('Type ''run_bfnt_a3_season1'' and press <RET>.\n');
