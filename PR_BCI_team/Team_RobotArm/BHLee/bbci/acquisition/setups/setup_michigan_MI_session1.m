global TODAY_DIR REMOTE_RAW_DIR VP_CODE CLSTAG VP_CAT

% TODO: decide study name

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

% TODO: VP_CAT = 2 or watch relax measurement of patients and then decide?
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

path([BCI_DIR 'acquisition/setups/michigan'], path);

startup_new_bbci_online

fprintf('\n\nWelcome to BBCI Experiment\n\n');

system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
bvr_sendcommand('stoprecording');

%% Load Workspace into the BrainVision Recorder
% TODO: create BV workspace
bvr_sendcommand('loadworkspace', 'ActiCap_32ch_motor_dense');

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

acq_makeDataFolder('log_dir',1,'multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

LOG_DIR = [TODAY_DIR '\log\'];

all_classes= {'left', 'right', 'foot'};
  
%% bbci_default
bbci= [];
[dmy, bbci.subdir]= fileparts(TODAY_DIR(1:end-1));
bbci.clab= {'F3-4','FC5-6','C5-6','CP5-6','P3-4','POz'};

% TODO: decide whether patch or laplacian
cfy_name= ['patches_C3z4_' patch '_' bandstr '_*.mat'];
bbci.classes= 'auto';
bbci.classDef= {1, 2, 3; 'left','right','foot'};

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

acqFolder = [BCI_DIR 'acquisition/setups/michigan/'];
pyff_fb_setup = [TODAY_DIR 'pyff_feedback_setup'];

VP_SCREEN= [-2250 -300 1920 1200];

fprintf('Type ''run_michigan_MI_session1'' and press <RET>.\n');