% 1. Vormessungen (checkerboard, artifacts, relax, CNV, oddball)
addpath([BCI_DIR 'acquisition/setups/season10']);
acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
VEP_file = [acqFolder 'VEP_feedback'];
CNV_file = [acqFolder 'CNV_feedback'];
CENTERSPELLER_file = [acqFolder 'CenterSpeller_feedback'];
ODDBALL_file = [acqFolder 'Oddball_feedback'];
RUN_END = {'S246' 'S247' 'S255'}; % check which one is actually the final marker

%% Relaxation
addpath([BCI_DIR 'acquisition/setups/season10']);
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
stimutil_waitForInput('msg_next','to start RELAX recording.');
stim_artifactMeasurement(seq, wav, opt);
close all

%% -- Eye Movements --
%[seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
%stimutil_waitForInput('msg_next','to start EYE-CALIBRATION practice');
%stim_artifactMeasurement(seq, wav, opt, 'test',1);
%[seq, wav, opt]= setup_season10_artifacts('clstag', '');
%stimutil_waitForInput('msg_next','to start EYE-CALIBRATION recording');
%stim_artifactMeasurement(seq, wav, opt);
%close all

%% Pyff starten
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
%pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',1, 'bvplugin', 0);

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
pyff('play', 'basename', 'VEP_', 'impendances', 0)
stimutil_waitForMarker(RUN_END);
fprintf('VEP recording finished.\n')
pyff('quit');

%% CNV - Practice
pyff('init', 'VisualOddballVE_CNV'); pause(.5);
pyff('load_settings', CNV_file);
pyff('setint','nTrials',10);
stimutil_waitForInput('msg_next','to start CNV practice.');
pyff('play');
stimutil_waitForMarker(RUN_END); % ??
fprintf('CNV practice finished.\n')
pyff('quit');

%% CNV - Recording
pyff('init', 'VisualOddballVE_CNV'); pause(.5);
pyff('load_settings', CNV_file);
stimutil_waitForInput('msg_next','to start CNV recording.');
pyff('play', 'basename', 'CNV_', 'impedances', 0);
stimutil_waitForMarker(RUN_END);
fprintf('CNV recording finished.\n')
pyff('quit');
fprintf('Press <RETURN> to continue.\n'); pause;

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
pyff('play', 'basename', 'Oddball', 'impedances', 0);
stimutil_waitForMarker(RUN_END);
fprintf('Oddball recording finished.\n')
pyff('quit');
fprintf('Press <RETURN> to continue.\n'); pause;


%% Visual Speller
condition_tags= {'','FixedSequence'};
order= perms(1:length(condition_tags));
conditionsOrder= uint8(order(1+mod(VP_NUMBER-1, size(order,1)),:));

phrase_practice= 'BCI';
phrase_calibration= 'BRAIN_COMPUTER_INTERFACE';
phrase_copyspelling= 'LET_YOUR_BRAIN_TALK.';

%% main loop
for jj= conditionsOrder,
  speller_name= ['CenterSpeller' condition_tags{jj}];
  switch(condition_tags{jj}),
    case '',
      rand_seq= true;
    case 'FixedSequence',
      rand_seq= false;
  end
  
  %% Practice
  msg= sprintf('to start %s ', condition_tags{jj});
  stimutil_waitForInput('msg_next', [msg 'practice']);
  setup_speller
  % practice
  pyff('set','desired_phrase',phrase_practice)
  pyff('setint', 'offline',1);
  pyff('set', 'randomize_sequence', rand_seq);
  pyff('play');
  stimutil_waitForMarker(RUN_END,'verbose',1);
  pyff('quit');
  
  %% Calibration
  stimutil_waitForInput('msg_next', [msg 'calibration']);
  setup_speller
  pyff('set', 'desired_phrase',phrase_calibration)
  pyff('setint', 'offline',1);
  pyff('set', 'randomize_sequence', rand_seq);
  pyff('save_settings', speller_name);
  pyff('play', 'basename', ['calibration_' speller_name], 'impedances', 0);
  stimutil_waitForMarker(RUN_END);
  pyff('quit');    
  
  %% Train the classifier
  bbci= bbci_default;
  bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, '_', VP_CODE);
  bbci_bet_prepare
  feedback_settings= pyff_loadSettings(CENTERSPELLER_file);
  bbci.setup_opts.nr_sequences= feedback_settings.nr_sequences;
  bbci_bet_analyze
  fprintf('Type ''dbcont'' to continue\n');
  keyboard
  bbci_bet_finish
  close all
  
  %% Online copy-spelling
  stimutil_waitForInput('msg_next', [msg 'copy-spelling']);
  setup_speller
  pyff('set', 'desired_phrase',phrase_copyspelling);
  pyff('setint', 'offline',0);
  pyff('set', 'randomize_sequence', rand_seq);
  pyff('play', 'basename', ['copy_' speller_name], 'impedances',0);
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
  pyff('quit');
  
  %% Freespelling
  stimutil_waitForInput('msg_next', [msg 'free-spelling']);
  setup_speller
  pyff('set', 'desired_phrase','');
  pyff('setint', 'offline',0);
  pyff('set', 'copy_spelling', false);
  pyff('set', 'randomize_sequence', rand_seq);
  pyff('play', 'basename', ['free_' speller_name], 'impedances', 0);
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Free-spelling run finished.\n')
  pyff('quit');
end

if ~strcmp(VP_CODE, 'Temp');
  save(VP_COUNTER_FILE, 'VP_NUMBER');
end
