setup_bbci_online; %% needed for acquire_bv

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'BrainCap_64_motor_dense');      %% Berlin Fast'n'Easy Cap
try,
  bvr_checkparport('type','S');
catch
  error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
end

global TODAY_DIR
acq_getDataFolder;
