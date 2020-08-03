speller= 'CenterSpeller';
%speller= 'HexoSpeller';
%speller= 'CakeSpeller';

acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
SPELLER_SETUP_FILE = [acqFolder speller '_feedback'];
RUN_END = {'S246' 'S255'}; % check which one is actually the final marker

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);

phrase_practice= 'BCI';
phrase_calibration= 'GEDANKENKRAFT';
phrase_copyspelling= 'DIE_BERLINER_LUFT';
%phrase_calibration= 'BRAIN_COMPUTER_INTERFACE';
%phrase_copyspelling= 'LET_YOUR_BRAIN_TALK';

msg= ['to start ' speller ' '];

%% Practice
setup_speller
stimutil_waitForInput('msg_next', [msg 'practice']);
pyff('set', 'desired_phrase',phrase_practice)
pyff('setint', 'offline',1);
pyff('setint', 'nCountdown', 10);
pyff('play');
stimutil_waitForMarker(RUN_END, 'verbose',1);
pyff('quit');

%% Calibration
setup_speller
stimutil_waitForInput('msg_next', [msg 'calibration']);
pyff('set', 'desired_phrase',phrase_calibration)
pyff('setint', 'offline',0);
pyff('save_settings', speller); pause(0.5);
pyff('play', 'basename', ['calibration_' speller], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');    

%% Train the classifier
feedback_settings= pyff_loadSettings(SPELLER_SETUP_FILE);
bbci.calibrate.file= strcat('calibration_*');
bbci.calibrate.save.file= strcat('bbci_classifier_', speller, VP_CODE);
bbci.calibrate.settings.nSequences= feedback_settings.nr_sequences;

%%%%%% BEGIN ONLY FOR TESTING
%bbci.calibrate.settings.reject_artifacts = 0;
%bbci.calibrate.settings.reject_channels = 0;
%%%%%% END ONLY FOR TESTING

[bbci, data]= bbci_calibrate(bbci);
bbci= copy_subfields(bbci, bbci_default);
bbci_save(bbci, data);

%% Freespelling
setup_speller
stimutil_waitForInput('msg_next', [msg 'free-spelling']);
pyff('set', 'desired_phrase','');
pyff('setint', 'offline',0);
pyff('set', 'copy_spelling', false);
pyff('play', 'basename', ['free_' speller], 'impedances', 0);
pause(1)
bbci_apply(bbci);
%% To stop the recording: type 'ppTrigger(255)' in a second Matlab
fprintf('Free-spelling run finished.\n')
pyff('quit');

%% Online Copy-spelling
stimutil_waitForInput('msg_next', [msg 'copy-spelling']);
setup_speller;
pyff('set', 'desired_phrase',phrase_copyspelling);
pyff('setint', 'offline',0);
pyff('play', 'basename', ['copy_' speller], 'impedances',0);
bbci_apply(bbci);
fprintf('Copy-spelling run finished.\n')
pyff('quit');
