%pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
pyff('startup','a',['C:\Dokumente und Einstellungen\ml\Eigene Dateien\stephan_luis'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);

feedback= 'ImageCreator';
msg= ['to start ' feedback ' '];


% %% Calibration
stimutil_waitForInput('msg_next', [msg 'calibration']);
setup_training_feedback
pause(7)
pyff('save_settings', ['calibration_' feedback]);
pyff('play', 'basename', ['calibration_' feedback], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');

%% Train the classifier
bbci.calibrate.file= strcat('calibration_*');
bbci.calibrate.save.file= strcat('bbci_classifier_', feedback, VP_CODE);

%%%%%% BEGIN ONLY FOR TESTING
if strcmp(VP_CODE, 'VPtemp'),
  bbci.calibrate.settings.reject_artifacts = 0;
  bbci.calibrate.settings.reject_channels = 0;
end
%%%%%% END ONLY FOR TESTING

[bbci, data]= bbci_calibrate(bbci);
bbci= copy_subfields(bbci, bbci_default);
bbci_save(bbci, data);


%% copy-task
stimutil_waitForInput('msg_next', [msg 'copy task']);

% bbci_apply_close(bbci); pyff('quit');
setup_painting_feedback
pyff('save_settings', ['copy_' feedback]);
pause(2);%0.5);
feedback_settings= pyff_loadSettings([TODAY_DIR 'copy_' feedback]);
bbci.control.param{1}.nSequences= feedback_settings.n_groups;
pyff('play', 'basename', ['copy_' feedback], 'impedances', 0);
pause(1)
bbci_apply(bbci);
pyff('quit');

