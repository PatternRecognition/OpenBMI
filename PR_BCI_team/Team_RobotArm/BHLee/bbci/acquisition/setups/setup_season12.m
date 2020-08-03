if isempty(CLSTAG),
  error('Unknown classes, start with all 3 classes');
  CLSTAG = 'LRF';
end
if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

if ~exist(PATCH,'var')
  warning('PATCH undefined - assuming patch twelve');
  PATCH = 'twelve';
end

band = [8 32];
bandstr= strrep(sprintf('%g-%g', band'),'.','_');

path([BCI_DIR 'acquisition/setups/season12'], path);

%% TODO: startup_new_bbci_online when ready
setup_bbci_online;

fprintf('\n\nWelcome to BBCI Season 12\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
%% TODO: create ActiCap_64ch_motor_dense
bvr_sendcommand('loadworkspace', 'ActiCap_64ch_motor_dense');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR REMOTE_RAW_DIR

acq_makeDataFolder('log_dir',1);

REMOTE_RAW_DIR= TODAY_DIR;

LOG_DIR = [TODAY_DIR '\log\'];

all_classes= {'left', 'right', 'foot'};
ci1= find(CLSTAG(1)=='LRF');
ci2= find(CLSTAG(2)=='LRF');

cfy_name= ['cspp_C3z4_' PATCH '_' bandstr '_' CLSTAG '.mat'];
copyfile([EEG_RAW_DIR '/subject_independent_classifiers/season12/' cfy_name], TODAY_DIR);

%% Adaptation
adaptation.active= 1;
adaptation.fcn= @bbci_adaptation_pcovmean;
adaptation.param= {struct('ival',[500 4000])};
adaptation.filename= ['$TMP_DIR/bbci_classifier_cspp_C3z4_' PATCH '_' bandstr '_' CLSTAG '_pmean'];
adaptation.log.output= 'screen';

%% make bbci_Default available in the run_script
global adaptation_default
adaptation_default= adaptation;

%% TODO: these lines go now in bbci or just in bbci.setup_opts?
bbci.classes = all_classes([ci1 ci2]);
bbci.classDef= cat(1, {1, 2}, bbci.classes);


fprintf('Type ''run_season12'' and press <RET>.\n');

%VP_SCREEN= [-799 0 800 600];
VP_SCREEN= [-1279 900-1024+1 1280 1024-19];
