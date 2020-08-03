session_name= 'labrotation2012_StephanGabler_LuisSeoane';

fprintf('\n\nWelcome to the study "%s"!\n\n', session_name);
startup_new_bbci_online;
addpath([BCI_DIR 'acquisition/setups/' session_name]);

%% Start BrainVisionn Recorder, load workspace and check triggers
system('start Recorder'); pause(1);
bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', session_name);
try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end


%% Create data folder
global TODAY_DIR

acq_makeDataFolder('multiple_folders', 1);

RUN_END= 255;

%% Define some variables
clear bbci_default
bbci_default.source.acquire_fcn= @bbci_acquire_bv;
bbci_default.source.acquire_param= {struct('fs', 100)};
bbci_default.log.output= 'screen&file';
bbci_default.log.classifier= 1;
bbci_default.control.fcn= @bbci_control_ImageCreator;
bbci_default.feedback.receiver = 'pyff';
bbci_default.quit_condition.marker= RUN_END;

BC= [];
BC.folder= TODAY_DIR;
BC.read_param= {'fs',100};
BC.marker_fcn= @mrk_defineClasses;
BC.marker_param= {{[101:200], [1:100]; 'target', 'nontarget'}};
BC.save.file= 'bbci_classifier_ImageCreator';
BC.fcn= @bbci_calibrate_ERP_Speller;
BC.settings= struct('cfy_clab', '*', ...
                    'nClasses', 6, ...
                    'nSequences', 10, ...
                    'cue_markers', cat(2, BC.marker_param{1}{1,:}));
BC.save.figures=1;
bbci= struct('calibrate', BC);

% Display feedback on laptop screen
screen_pos= get(0, 'ScreenSize');
VP_SCREEN= [0 0 screen_pos(3:4)];

bvr_sendcommand('viewsignals');

fprintf(['run_' session_name '\n']);

