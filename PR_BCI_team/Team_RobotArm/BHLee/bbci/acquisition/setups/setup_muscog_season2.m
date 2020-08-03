if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/stimulation/muscog_season2'], path);
path([BCI_DIR 'acquisition/setups/muscog_season2'], path);
path([BCI_DIR 'acquisition/setups/bfnt_a3_season1'], path);
path([BCI_DIR 'acquisition/setups/season10'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to MusCog - Season 2 (Irene)\n\n');

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

VP_SCREEN= [-1919 0 1920 1200];
%VP_SCREEN= [-1023 0 1024 768];
fprintf('Type ''run_muscog_season2'' or ''run_muscog_season2_plus_bfnt_a3'' and press <RET>.\n');
