% Experiment investigating covert attention shifts 
% Bimodal measurement (EEG/NIRS) over occipital areas.

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

% Set automatic subject counter
if strcmp(VP_CODE, 'Temp');
  vp_number= 1;
else
  vp_counter_file= [DATA_DIR 'alpha_NIRS_Counter'];
  % delete([vp_counter_file '.mat']);   %% for reset
  if exist([vp_counter_file '.mat']),
    load(vp_counter_file, 'vp_number');
  else
    vp_number= 0;
  end
  vp_number= vp_number + 1;
  fprintf('VP number %d.\n', vp_number);
end

%
addpath([BCI_DIR 'acquisition/setups/project10_alpha']);
addpath([BCI_DIR 'acquisition/setups/season10']);

fprintf('\n\nWelcome to alpha covert attention experiment using EEG/NIRS.\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'EasyCap_26ch_occipital');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('multiple_folders', 1);
%mkdir([TODAY_DIR 'data']);
VP_SCREEN = [1920 0 1280 1024];
%VP_SCREEN = [0 0 1280 1024]; % eyetracker Bildschirm

fprintf('Type ''run_project10_alpha'' and press <RET>\n');

%% Feedback settings
general_port_fields.feedback_receiver= 'pyff';

VP_SCREEN = [0 0 1280 1024];
