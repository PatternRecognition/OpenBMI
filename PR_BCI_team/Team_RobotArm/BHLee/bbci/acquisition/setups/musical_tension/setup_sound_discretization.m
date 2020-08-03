if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end



setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to the Experiment\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'musical_tension_new');
%bvr_sendcommand('loadworkspace', 'musical_tension_no_eeg');

%bvr_sendcommand('loadworkspace', 'FastnEasy_auditory_P300');
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR

acq_makeDataFolder;

pos= get(0,'ScreenSize');
%VP_SCREEN= [pos(3) 0 1920 1200];
VP_SCREEN= [-pos(3) 0 1920 1200];
%fprintf('Type ''run_bfnt_a3_season1'' and press <RET>.\n');

