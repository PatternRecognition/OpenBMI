% Behavioral study 
%

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

% Subject counter
if strcmpi(VP_CODE, 'Temp');
  vp_number= 1;
else
  vp_counter_file= [DATA_DIR 'RSVP_online_VP_Counter'];
  % delete([vp_counter_file '.mat']);   %% for reset
  if exist([vp_counter_file '.mat']),
    load(vp_counter_file, 'vp_number');
  else
    vp_number= 0;
  end
  vp_number= vp_number + 1;
  fprintf('VP number %d.\n', vp_number);
end

% Add directories
addpath([BCI_DIR 'acquisition/setups/RSVP']);
addpath([BCI_DIR 'acquisition/setups/RSVP/behavioral']);

%%
fprintf('\n\nWelcome to RSVP *** BEHAVIORAL ***\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'one_channel.rwksp');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
mkdir([TODAY_DIR 'data']);

% VP_SCREEN = [0 0 1920 1200];
fprintf('Type ''run_pilot'' and press <RET>\n');



