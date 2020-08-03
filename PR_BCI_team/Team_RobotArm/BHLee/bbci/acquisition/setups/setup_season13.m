global TODAY_DIR REMOTE_RAW_DIR VP_CODE CLSTAG VP_CAT

if isempty(CLSTAG)
  warning('CLSTAG undefined, starting with 40 trials per class combination');
end

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end
if isempty(VP_CAT),
  warning('VP_CAT undefined - assuming CAT 2');
  VP_CAT= 2;
end
if ~exist('patch','var')
  warning('patch undefined - assuming patch small');
  patch = 'small';
end

band = [8 32];
bandstr= strrep(sprintf('%g-%g', band'),'.','_');

path([BCI_DIR 'acquisition/setups/season13'], path);

startup_new_bbci_online_season13

fprintf('\n\nWelcome to BBCI Season 13\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
% 2 ist wit C4 wrong
bvr_sendcommand('loadworkspace', 'ActiCap_62ch_motor_dense_right_mastoid2');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

acq_makeDataFolder('log_dir',1,'multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

LOG_DIR = [TODAY_DIR 'log\'];

all_classes= {'left', 'right', 'foot'};
  
%% bbci_default
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.clab= {'not','E*','Fp*','A*','FAF*','*9','*10','PO*','O*','T*'};

if ischar(CLSTAG)
  ci1= find(CLSTAG(1)=='LRF');
  ci2= find(CLSTAG(2)=='LRF');
  cfy_name= ['patches_C3z4_' patch '_' bandstr '_' CLSTAG '.mat'];
  bbci.classes= all_classes([ci1 ci2]);
  bbci.classDef= {1, 2; bbci.classes{:}};
else
  cfy_name= ['patches_C3z4_' patch '_' bandstr '_*.mat'];
  bbci.classes= 'auto';
  bbci.classDef= {1, 2, 3; 'left','right','foot'};
end
copyfile([EEG_RAW_DIR '/subject_independent_classifiers/season13/' cfy_name], TODAY_DIR);

%% Adaptation default
bbci.adaptation.active= 1;
bbci.adaptation.log.output= 'screen&file';

%% log default
bbci.log.output= 'screen&file';
bbci.log.classifier= 1;
bbci.log.force_overwriting= 1;

%% source default
bbci.source.acquire_fcn= @bbci_acquire_bv;
bbci.source.acquire_param = {struct('fs',100)};
bbci.source.marker_mapping_fcn= '';

bbci.feedback.receiver= 'pyff';
bbci.quit_condition.marker= 254;

%% make bbci_Default available in the run_script
global bbci_default
bbci_default= bbci;

acqFolder = [BCI_DIR 'acquisition/setups/season13/'];
pyff_fb_setup = [TODAY_DIR 'pyff_feedback_setup'];

VP_SCREEN= [-2250 -300 1920 1200];

fprintf('Type ''run_season13'' and press <RET>.\n');