acqFolder = [BCI_DIR 'acquisition/setups/special_event/' session_name '/'];
CENTERSPELLER_file = [acqFolder 'CenterSpeller_feedback'];
RUN_END = {'S246' 'S255'}; % check which one is actually the final marker

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);

phrase_practice= 'BCI';
phrase_calibration= 'BE_AWARE.';
phrase_copyspelling= 'DIE_GEDANKEN_SIND_FREI.';
speller= 'CenterSpeller';
msg= ['to start ' speller ' '];

%% Practice
stimutil_waitForInput('msg_next', [msg 'practice']);
setup_speller
pyff('set', 'desired_phrase',phrase_practice)
pyff('setint', 'offline',1);
pyff('play');
stimutil_waitForMarker(RUN_END, 'verbose',1);
pyff('quit');

%% Calibration
stimutil_waitForInput('msg_next', [msg 'calibration']);
setup_speller
pyff('set', 'desired_phrase',phrase_calibration)
pyff('setint', 'offline',1);
pyff('save_settings', speller);
pyff('play', 'basename', ['calibration_' speller], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');    

%% Train the classifier
feedback_settings= pyff_loadSettings(CENTERSPELLER_file);
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
stimutil_waitForInput('msg_next', [msg 'free-spelling']);
setup_speller
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
