path([BCI_DIR 'acquisition/setups/fasor_LCT2'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWillkommen bei der FaSor-Experimentreihe LCT2!\n\n');

% Load Workspace into the BrainVision Recorder: alpha or beta depending on
% head size!
try,
  bvr_sendcommand('loadworkspace', 'BrainCap_beta_64ch_EMGf_EOGhv');      %% 
  %bvr_sendcommand('loadworkspace', 'BrainCap_alpha_64ch_EMGf_EOGhv');      %% 
catch
    %error(sprintf('BrainVision Recorder could not find workspace %s', 'BrainCap_alpha_64ch_EMGf_EOGhv'));
    error(sprintf('BrainVision Recorder could not find workspace %s', 'BrainCap_beta_64ch_EMGf_EOGhv'));
end

try,
  bvr_checkparport('type','S');
catch
  error(sprintf('BrainVision Recorder must be running.\nThen restart %s.', mfilename));
end

global TODAY_DIR REMOTE_RAW_DIR
acq_getDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

VP_SCREEN = [0 0 1920 1200];

system('D:\Fasor\LCT_Markertool_V1002\LCTMarker.exe &');
fprintf('   VL: LCT Markertool testen und starten\n')
fprintf('   VL: ev. noch Bildschirm der VP zum primären Windows-Bildschirm machen (für LCT Simulator!)\n');
fprintf('   VL: run_fasor_LCT2 \n \n');