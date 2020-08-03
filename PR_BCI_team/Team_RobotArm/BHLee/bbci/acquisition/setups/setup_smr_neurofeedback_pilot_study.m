if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/smr_neurofeedback_season1'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to SMR Neurofeedback Pilot Study\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'reducerbox_64std_smr_pilot_study');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder('log_dir',1);

VP_SCREEN= [1920 0 1920 1200];
fprintf('Type ''run_smr_neurofeedback_pilot_study'' and press <RET>.\n');
