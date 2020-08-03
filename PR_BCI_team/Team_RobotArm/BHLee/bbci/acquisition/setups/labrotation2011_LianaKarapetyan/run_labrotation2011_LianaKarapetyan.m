% 1. Vormessungen (checkerboard, artifacts, relax, CNV, oddball)

acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
VEP_file = [acqFolder 'VEP_feedback'];
RSVP_file = [acqFolder 'RSVP_Color116_feedback'];
ODDBALL_file = [acqFolder 'Oddball_feedback'];
RUN_END = {'S246' 'S247' 'S255'}; % check which one is actually the final marker

%% Start TOBI Signal Server
start_signalserver('server_config_visualERP_gSAHARA.xml'); pause(1);
cmd= [BCI_DIR 'online\communication\signalserver\Scope\TOBI_RemoteScope.exe &'];
system(cmd);
opt_rec= struct('quit_marker',255, 'position',[-2555 920 155 80]);
opt_rec1= struct('quit_marker',253, 'position',[-2555 920 155 80]);
fprintf('In TiA Scope, press ''connect'', choose TCP and press ''Receive data.\n');
% For working with g.tec amps
global acquire_func; 
acquire_func= @acquire_sigserv;

%% Relaxation
addpath([BCI_DIR 'acquisition/setups/season10']);
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
seq_test= strrep(seq, '[10]', '[2]');
stimutil_waitForInput('msg_next','to start RELAX practice.');
stim_artifactMeasurement(seq_test, wav, opt, 'test',1,'bv_host','');
stimutil_waitForInput('msg_next','to start RELAX recording.');
signalServer_startrecoding(['relax' VP_CODE], opt_rec1); pause(3);
stim_artifactMeasurement(seq, wav, opt, 'test',1,'bv_host','');
close all

%% Pyff starten
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
%pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',1, 'bvplugin', 0);
send_xmlcmd_udp('init', '127.0.0.1', 12345);
fprintf('Starting Pyff...\n'); pause(10);

%% VEP checkboard - practice
pyff('init','CheckerboardVEP'); pause(.5)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
pyff('setint','nStim',10);
stimutil_waitForInput('msg_next','to start VEP practice.');
pyff('play');
stimutil_waitForMarker(RUN_END);
fprintf('VEP practice finished.\n')
pyff('quit');

%% VEP checkboard - recording
pyff('init','CheckerboardVEP'); pause(.5);
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
stimutil_waitForInput('msg_next','to start VEP recording.');
signalServer_startrecoding(['VEP_' VP_CODE], opt_rec); pause(3);
pyff('play');
stimutil_waitForMarker(RUN_END);
fprintf('VEP recording finished.\n')
pyff('quit');
 

%% Oddball - Practice
pyff('init', 'VisualOddballVE'); pause(.5);
pyff('load_settings', ODDBALL_file);
pyff('setint','nTrials',10);
stimutil_waitForInput('msg_next','to start Oddball practice.');
pyff('play');
stimutil_waitForMarker(RUN_END);
fprintf('Oddball practice finished.\n')
pyff('quit');

%% Oddball - Recording
pyff('init', 'VisualOddballVE'); pause(.5);
pyff('load_settings', ODDBALL_file);
stimutil_waitForInput('msg_next','to start Oddball recording.');
signalServer_startrecoding(['Oddball_' VP_CODE], opt_rec); pause(3);
pyff('play');
stimutil_waitForMarker(RUN_END);
fprintf('Oddball recording finished.\n')
pyff('quit');
fprintf('Press <RETURN> to continue.\n'); pause;


%% RSVP Speller
soa= 116;
tag= ['Color' int2str(soa) 'ms'];
speller= ['RSVP_Color' int2str(soa) 'ms'];

phrase_practice= choose_case('RSVP!');
phrase_calibration= choose_case('BRAIN-COMPUTER-INTERFACE');
phrase_copyspelling= choose_case('LET-YOUR-BRAIN-TALK.');

TRIAL_COUNT= 1;
TRIAL_YESNO= 2;
TRIAL_CALIBRATION= 3;
TRIAL_FREESPELLING= 4;
TRIAL_COPYSPELLING= 5;

%% Practice
stimutil_waitForInput('msg_next', ['to start ' tag ' practice']);
setup_RSVP_feedback;
pyff('setint', 'trial_type',TRIAL_CALIBRATION);
pyff('setint','show_alphabet',0);
pyff('set', 'words',{phrase_practice});
pause(.01)
fprintf('Ok, starting...\n');
pause(5)
pyff('play');
stimutil_waitForMarker(RUN_END,'verbose',1);
pyff('quit');

%% Calibration
stimutil_waitForInput('msg_next', ['to start ' tag ' calibration']);
setup_RSVP_feedback
pyff('setint','trial_type',TRIAL_CALIBRATION);
pyff('setint','show_alphabet',0);
pyff('set', 'words',{phrase_calibration});
fprintf('Ok, starting...\n');
pyff('save_settings', ['calibration_' speller]);
signalServer_startrecoding(['calibration_' speller '_' VP_CODE], opt_rec);
pause(3);
pyff('play');
pause(5)
stimutil_waitForMarker(RUN_END);
pyff('quit');

%% Train the classifier
feedback_settings= pyff_loadSettings(RSVP_file);
bbci.calibrate.file= strcat('calibration_', speller, '_', VP_CODE);
bbci.calibrate.save.file= strcat('bbci_classifier_', speller, '_', VP_CODE);
bbci.calibrate.settings.ref_ival= [-soa 0];
bbci.calibrate.settings.nClasses= 30;
bbci.calibrate.settings.nSequences= feedback_settings.sequences_per_trial;
[bbci, data]= bbci_calibrate(bbci);
fprintf('Type dbcont to continue\n');
keyboard

bbci.source.acquire_fcn= @bbci_acquire_sigserv;
%bbci.source.min_blocklength= 10;
bbci.control.fcn= @bbci_control_RSVP_Speller;
bbci.control.condition.marker= [31:100, 199];
bbci.feedback.receiver = 'pyff';
bbci.quit_condition.marker= 255;
bbci.log.output= 'screen&file';
bbci.log.file= ['bbci_log_' speller '_' VP_CODE];
bbci.log.classifier= 1;
bbci.source.log.output= 'file';
bbci_save(bbci, data);
close all

%% Freespelling - testing
stimutil_waitForInput('msg_next', ['to TEST ' tag ' free-spelling']);
setup_RSVP_feedback
pyff('setint', 'trial_type',TRIAL_FREESPELLING);
pyff('set','words','');
pyff('set','show_alphabet',1);
pyff('set','show_trial_countdown',1);
fprintf('Ok, starting...\n');
pyff('play');
pause(1)
bbci_apply(bbci);
% To stop the recording: press a button that generates R2 or R4
% or type 'ppTrigger(2)' in a second Matlab
fprintf('Free-spelling test finished.\n')
pyff('quit');

%% Freespelling
stimutil_waitForInput('msg_next', ['to start ' tag ' free-spelling']);
setup_RSVP_feedback
pyff('setint', 'trial_type',TRIAL_FREESPELLING);
pyff('set','words','');
pyff('set','show_alphabet',1);
pyff('set','show_trial_countdown',1);
fprintf('Ok, starting...\n');
pyff('save_settings', ['free_' speller]);
eegfile= signalServer_startrecoding(['freespelling_' speller '_' VP_CODE], opt_rec); 
bbci= bbci_log_setHeaderInfo(bbci, ['# EEG file: ' eegfile]);
pause(3);
pyff('play');
pause(1)
bbci_apply(bbci);
%% To stop the recording: press a button that generates R2 or R4
%% or type 'ppTrigger(255)' in a second Matlab
fprintf('Free-spelling run finished.\n')
pyff('quit');

%% Online copy-spelling
stimutil_waitForInput('msg_next', ['to start ' tag ' copy-spelling']);
setup_RSVP_feedback
pyff('setint', 'trial_type',TRIAL_COPYSPELLING); 
pyff('set', 'words',{phrase_copyspelling});
pyff('set', 'present_word_time',0);
fprintf('Ok, starting...\n');
pyff('save_settings', ['copy_' speller]);
eegfile= signalServer_startrecoding(['copyspelling_' speller '_' VP_CODE], opt_rec);
bbci= bbci_log_setHeaderInfo(bbci, ['# EEG file: ' eegfile]);
pause(3);
pyff('play');
pause(1);
bbci_apply(bbci);
fprintf('Copy-spelling run finished.\n')
pyff('quit');
 
fprintf('Experiment finished!\n');
