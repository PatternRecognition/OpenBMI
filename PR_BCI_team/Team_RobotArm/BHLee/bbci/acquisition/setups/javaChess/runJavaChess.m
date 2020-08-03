%% set everything up
warning('Is the proper VP_CODE set?');
input('');

global general_port_fields TODAY_DIR REMOTE_RAW_DIR SESSION_TYPE DATA_DIR VP_CODE

% addpath([BCI_DIR '\acquisition\stimulation\photobrowser']);

acq_makeDataFolder('multiple_folders',1);

REMOTE_RAW_DIR= TODAY_DIR;

set_general_port_fields('localhost');
general_port_fields.feedback_receiver = 'matlab';

global VP_SCREEN;
position = [-1920 0 1920 1200];
% small screen for testing
%warning('using VP_SCREEN for testing');

% start the recorder and load a workspace
% system('c:\Vision\Recorder\Recorder.exe &'); pause(1);
% % % bvr_sendcommand('stoprecording');
bvr_sendcommand('loadworkspace', ['reducerbox_64std_visual']);
    
%% Calibration data
clc;
disp('Recording started. Please start the calibration phase by hand.\n');
bvr_startrecording(['javaChess_train_' VP_CODE], 'impedances', 0);


%% run the analysis
setup_javaChess_online; edit bbci_bet_analyze_AEP.m;
bbci_bet_prepare;
bbci_bet_analyze;
bbci_bet_finish;


%% Online data
clc;
disp('Recording started....\n');
bvr_startrecording(['javaChess_train_' VP_CODE], 'impedances', 0);
disp('Classifier started.... Please start application by hand.\n');
bbci_bet_apply({[TODAY_DIR '\bbci_classifier'});

