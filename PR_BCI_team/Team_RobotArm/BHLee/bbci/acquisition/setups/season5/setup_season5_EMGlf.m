setup_bbci_bet_unstable; %% needed for acquire_bv
% First of all check that the parllelport is properly conneted.

fprintf('\n\nWelcome to Season 5\n\n');
try,
  bvr_checkparport('type','S');
catch,
  fprintf('BrainVision Recorder must be running!\nStart it and rerun %s.\n\n', mfilename)
  return;
end

%% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'EasyCap_128_EMGlf_EOGv');
bvr_sendcommand('viewsignals');

acq_getDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

bbci= [];
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'imag_*');
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
bbci.classDef= {1, 2; 'left', 'foot'};
bbci.classes= {'left','foot'};
bbci.feedback= '1d';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
bbci.setup_opts.usedPat= 'auto';

fprintf('\n\nClasses used: %s %s\n\n', bbci.classes{1}, bbci.classes{2});
close


