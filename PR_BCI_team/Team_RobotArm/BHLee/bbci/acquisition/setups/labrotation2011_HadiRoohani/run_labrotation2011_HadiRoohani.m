%% Trigger test
fprintf('Start NIRStar, press "Start" and go to "Pre scan". Then press RETURN. Can you see the triggers?')
pause
fprintf('\n')
for ii=[1 2 4 8 3 5 10 12 15];
  ppTrigger(ii)
  pause(.5)
end

%% Settings
general_port_fields= struct('bvmachine','127.0.0.1',...
                            'control',{{'127.0.0.1',12471,12487}},...
                            'graphic',{{'',12487}});
general_port_fields.feedback_receiver= 'pyff';

%% Relaxation
fprintf('\n\nRelax recording.\n');
[seq,wav,opt]= setup_season10_relax;
opt = rmfield(opt,'handle_cross');
seq= ['P2000 F21 P3000 F14P300000 F1P5000 F15P300000 F1P5000 F20P1000'];

fprintf('Press <RETURN> to start RELAX measurement.\n');
pause; fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt,'filename',[]);

%% Experimental design
offlineFile = 'quiz_offline';
onlineFile = 'quiz_online';
practiceFile = 'quiz_practice';

nirsdir = 'C:\Data';

% RUN_END = 8;

VEP_file = [sessiondir 'VEP_feedback'];
QUIZ_SETTINGS_file = [sessiondir 'VisualQuizSettings'];

%% Pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks\'],'gui',0);

%% Vormessung: VEP
pyff('init','CheckerboardVEP');pause(1)
pyff('load_settings', VEP_file);
pyff('setint','screen_pos',VP_SCREEN);
fprintf('Press <RETURN> to start VEP measurement.\n'); pause;
fprintf('Ok, starting...\n'),close all
%pyff('setdir', '');
pyff('setdir', 'basename', '');
pause(5)
pyff('play')
pause(5)
stimutil_waitForMarker(RUN_END);
fprintf('VEP measurement finished.\n')
bvr_sendcommand('stoprecording');
pyff('stop');
pyff('quit');
fprintf('Press <RET> to continue.\n'); pause;

%% Instruction and questionnaire  **** TODO
pyff('init','HTMLViewer');pause(1)
pyff('set','pages',{'SubjectInfo.htm' [sessiondir 'instruction.htm']});
pyff('play')

%% Practice
pyff('init','VisualQuiz');pause(1)
pyff('load_settings', QUIZ_SETTINGS_file);
pyff('set','quizFile',practiceFile);
pyff('setint','geometry',VP_SCREEN);
pyff('setint','online',0);
pyff('play'); pause(5)

% stimutil_waitForMarker('stopmarkers',RUN_END);
pyff('stop');
pyff('quit');
fprintf(['Practice finished!\n'])

fprintf('\n****************\nDON''T FORGET TO PRESS THE PLAY + RECORDING BUTTON BEFORE STARTING THE CALIBRATION!!\n****************\n')

%% *** CALIBRATION ### DONT FORGET TO RECORD !!!
pyff('init','VisualQuiz');pause(1)
pyff('load_settings', QUIZ_SETTINGS_file);
pyff('set','quizFile',offlineFile);
pyff('setint','nTrials',120);
pyff('setint','online',0);
pyff('setint','geometry',VP_SCREEN);

pyff('play'); pause(5)

stimutil_waitForMarker('stopmarkers',RUN_END);
pyff('stop');
pyff('quit');
fprintf('Practice finished!\n')

%% Calibrate/Train the classifier
bbci.calibrate.file= ['VisualQuiz' VP_CODE];
bbci.calibrate.save.file=  ['bbci_classifier_VisualQuiz' VP_CODE];
bbci.calibrate.settings.nClasses= 3;
bbci.log.file= ['bbci_log_VisualQuiz' VP_CODE];
warning off
[bbci, data]= bbci_calibrate(bbci);
warning on
bbci_save(bbci, data);

%% *** ONLINE ***
close all
pyff('setint','online',1);
pyff('set','tasks',bbci.classifier.classes)

[bbci, data]= bbci_apply(bbci);
% bbci_apply_close(bbci, data);

%%
fprintf(['Experiment finished!\n'])