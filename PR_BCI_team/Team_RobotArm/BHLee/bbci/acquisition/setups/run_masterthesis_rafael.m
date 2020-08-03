session_name= 'vitalbci_season2';

fprintf('\n\nWelcome to the study "%s"!\n\n', session_name);
startup_new_bbci_online;
HOST = 'localhost';
send_udp_xml('init', HOST, 12345);
addpath([BCI_DIR 'acquisition/setups/' session_name]);
addpath([BCI_DIR 'online/utils']);  %% do_set

%% Start BrainVisionn Recorder, load workspace and check triggers
if exist('c:\Vision\Recorder\Recorder.exe', 'file')
    system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
else
    system('D:\Vision\Recorder\Recorder.exe &'); pause(1);
end
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', session_name);
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

%% Create data folder
global TODAY_DIR
acq_makeDataFolder;
%VP_NUMBER= acq_vpcounter(session_name, 'new_vp');

%% Define some variables
classDef= {1, 2, 3;
           'left', 'right', 'foot'};

clear bbci_default
bbci_default.source.acquire_fcn= @bbci_acquire_bv;
bbci_default.source.acquire_param= {struct('fs', 100)};
bbci_default.source.log.output= 'screen&file';
bbci_default.source.record_signals= 1;
bbci_default.log.output= 'file';
bbci_default.adaptation.log.output= 'screen&file';

% For run 1:
kickstart_cfy= [EEG_RAW_DIR ...
                'subject_independent_classifiers/masterthesis_rafael/' ...
                'kickstart_vitalbci_season2_C3CzC4_9-15_15-35'];
feedback_arrow_training.receiver= 'matlab';
feedback_arrow_training.fcn= @bbci_feedback_cursor_training;
feedback_arrow_training.opt= ...
    strukt('trigger_classifier_list', {{1,2},{1,3},{3,2}}, ...
           'trials_per_run', 40);

% For runs 2-
BC.folder= TODAY_DIR;
BC.read_param= {'fs',100};
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {classDef, 'removevoidclasses', 1};
BC.save.figures=1;
adapt_opt= struct('ival', [1500 4000]);

feedback_arrow.receiver= 'matlab';
feedback_arrow.fcn= @bbci_feedback_cursor;
feedback_arrow.opt= ...
    strukt('trigger_classes_list', classDef(2,:), ...
           'trials_per_run', 80);

%VP_SCREEN is defined in the startup file (e.g., startup_bbcilaptop07)

bvr_sendcommand('viewsignals');

cd([BCI_DIR 'acquisition/setups/masterthesis_Rafael']);
ACQLOCAL= struct('file', 'masterthesis_rafael_script.m');
acq_run_script_in_recovery_mode;