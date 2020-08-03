myname= mfilename;
session_name= myname(5:end);

fprintf('\n\nWelcome to the study "%s"!\n\n', session_name);
startup_new_bbci_online;
addpath([BCI_DIR 'acquisition/setups/' session_name]);

%% Start BrainVisionn Recorder, load workspace and check triggers
system('start Recorder'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', session_name);
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end


%% Create data folder
global TODAY_DIR
acq_makeDataFolder('log_dir',1);
LOG_DIR = [TODAY_DIR '\log\'];
%VP_NUMBER= acq_vpcounter(session_name, 'new_vp');

RUN_END= [246 255];

% Display feedback on laptop screen
screen_pos= get(0, 'ScreenSize');
VP_SCREEN= [0 0 screen_pos(3:4)];

bvr_sendcommand('viewsignals');

eval(['run_' session_name '_script']);
