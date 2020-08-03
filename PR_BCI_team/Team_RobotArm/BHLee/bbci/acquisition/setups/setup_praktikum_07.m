setup_bbci_bet_unstable; %% needed for acquire_bv
% First of all check that the parllelport is properly conneted.

fprintf('\n\nWelcome to Praktikum 2007\n\n');
try,
  bvr_checkparport('type','S');
catch,
  fprintf('BrainVision Recorder must be running!\nStart it and rerun %s.\n\n', mfilename)
  return;
end

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'EasyCap_128_EOG');
bvr_sendcommand('viewsignals');

acq_getDataFolder;

addpath([BCI_DIR 'stimulation/praktikum07_bci02']);

%stimutil_sendKeys2;
