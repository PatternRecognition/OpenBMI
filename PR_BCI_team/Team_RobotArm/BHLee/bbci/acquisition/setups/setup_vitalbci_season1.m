path([BCI_DIR 'acquisition/setups/vitalbci_season1'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to Vital-BCI Season 1\n\n');

%If you like to crash the whole computer, do this:
%system('c:\Vision\Recorder\Recorder.exe &')

%If matlab crashed before, BVR might still be in recording mode
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
if strcmp(general_port_fields.bvmachine, 'bbcipc'),
  bvr_sendcommand('loadworkspace', 'FastnEasy128_EOGhv_EMGlrf');  %% Tuebingen
else
%   bvr_sendcommand('loadworkspace', 'FastnEasy_EMGlrf');      %% Berlin's Fast'n'Easy Caps
  %bvr_sendcommand('loadworkspace', 'EasyCap_128_EMGlrf_EOG');     %% Berlin EasyCap
  %bvr_sendcommand('loadworkspace', 'eci_128ch_EMGlrf');     %% Berlin Fast'n'Easy Cap (obsolete)
  bvr_sendcommand('loadworkspace', 'one_channel');
end
try
  bvr_checkparport('type','S');
catch
  error('BrainVision Recorder must be running.\nThen restart %s.', mfilename);
end

global TODAY_DIR REMOTE_RAW_DIR
acq_getDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
bbci= [];
bbci.setup= 'cspauto';
bbci.train_file= strcat(subdir, '/imag_arrow*');
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
bbci.classDef= {1, 2, 3; 'left', 'right', 'foot'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
bbci.setup_opts.usedPat= 'auto';
%If'auto' mode does not work robustly:
%bbci.setup_opts.usedPat= [1:6];
