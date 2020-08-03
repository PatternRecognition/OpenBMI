if isempty(CLSTAG),
  error('Variable CLSTAG has to be defined');
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/adaptationstudy_season1'], path);

setup_bbci_online; %% needed for acquire_bv

fprintf('\n\nWelcome to Adaptation Study Season 1\n\n');

%If you like to crash the whole computer, do this:
system('c:\Vision\Recorder\Recorder.exe &'); pause(1);

%If matlab crashed before, BVR might still be in recording mode
bvr_sendcommand('stoprecording');

% Load Workspace into the BrainVision Recorder
bvr_sendcommand('loadworkspace', 'FastnEasy_motor_63ch');      %% Berlin's Fast'n'Easy Caps

try
  bvr_checkparport('type','S');
catch
  error('BrainAmps must be switched on.\nThen restart %s.', mfilename);
end

global TODAY_DIR REMOTE_RAW_DIR
acq_getDataFolder('log_dir',1);
REMOTE_RAW_DIR= TODAY_DIR;

[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
all_classes= {'left', 'right', 'foot'};
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
bbci= [];
bbci.setup= 'adaptationstudy_season1';
bbci.train_file= strcat(subdir, '/imag_fbarrow_LapC3z4_pcovmean*');
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*'};
bbci.classes= all_classes([ci1 ci2]);
bbci.classDef= cat(1, {1, 2}, bbci.classes);
bbci.feedback= '1d';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_lapcsp');
bbci.setup_opts.usedPat= 'auto';

fprintf('run_adaptationstudy_season1\n');
