global CLSTAG
CLSTAG= 'LR';
if isempty(CLSTAG),
  error('Variable CLSTAG has to be defined');
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

path([BCI_DIR 'acquisition/setups/setup_Brasil'], path);

fprintf('\n\nWelcome to BBCI\n\n');

system('C:\Vision\Recorder\Recorder.exe &'); pause(1);
% bvr_sendcommand('loadworkspace', 'ActiCap_64ch_motor_mundus');
bvr_sendcommand('loadworkspace', 'EEG4BCI_Braz');

bvr_sendcommand('stoprecording');

try
  bvr_checkparport('type','S','bv_host',general_port_fields.bvmachine  );
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR
acq_makeDataFolder;
%4 testing
% TODAY_DIR(end-1)='3';
REMOTE_RAW_DIR= TODAY_DIR;

%% prepare settings for classifier training
[dmy, subdir]= fileparts(TODAY_DIR(1:end-1));
all_classes= {'left', 'right', 'foot'};
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');
bbci= [];
bbci.setup= 'cspauto';
bbci.clab= {'not','E*','Fp*','AF*','PO*','O*'};
bbci.classes= all_classes([ci1 ci2]);
bbci.classDef= cat(1, {1, 2}, bbci.classes);
bbci.feedback= '1d';
bbci.train_file= strcat(subdir, '/imag_fbarrow*');
bbci.setup_opts.ilen_apply= 500;
bbci.adaptation.UC= 0.1;
bbci.adaptation.UC_mean= 0.075;
bbci.adaptation.UC_pcov= 0.001;
bbci.adaptation.delay= 1000;
bbci.adaptation.load_tmp_classifier= 1;
%% make bbci_default available in the run_script
global bbci_default
bbci_default= bbci;

copyfile([EEG_RAW_DIR 'subject_independent_classifiers/Lap_C3z4_bp_' CLSTAG '_v6*'], ...
 TODAY_DIR);

VP_SCREEN= [1280 280 1024 768];

edit run_BCI_Brasil
