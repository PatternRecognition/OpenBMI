% Reset random-generator seed to produce new random numbers
% Take cputime in ms as basis
rand('seed',cputime*1000)

%% Preparation of the EEG cap
% bvr_sendcommand('checkimpedances');
stimutil_waitForInput('msg_next','when finished preparing the cap.');
bvr_sendcommand('viewsignals');
pause(5);

%% Relax measurement: recording
fprintf('\n\nRelax recording.\n');
[seq, wav, opt]= setup_season10_relax;
stimutil_waitForInput('msg_next','to start RELAX measurement.');
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);
clear wav

% %% Artifacts
% [seq, wav, opt]= setup_season10_artifacts_demo('clstag', '');
% fprintf('Press <RETURN> to start TEST artifact measurement.\n');
% pause; fprintf('Ok, starting...\n');
% stim_artifactMeasurement(seq, wav, opt, 'test',1);
% [seq, wav, opt]= setup_season10_artifacts('clstag', '');
% fprintf('Press <RETURN> to start artifact measurement.\n');
% pause; fprintf('Ok, starting...\n');
% stim_artifactMeasurement(seq, wav, opt);

%% ** Startup pyff **
close all
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0);
pause(4)

condition_tags= {'NoColor116ms', 'Color116ms', 'Color83ms'};
order= perms(1:length(condition_tags));
conditionsOrder= order(1+mod(VP_NUMBER-1, size(order,1)),:);

phrase_practice= choose_case('QUARZ');
phrase_calibration= choose_case('BRAIN_COMPUTER');
phrase_copyspelling2= choose_case('LET_YOUR_BRAIN_TALK.');
phrase_copyspelling1= choose_case('WINTER_IS_DEPRESSING');
phrase_copyspelling3= choose_case('DONT_WORRY_BE_HAPPY!');

%Trial type in RSVP speller
% 1: Count, 2: YesNo, 3: Calibration, 4: FreeSpelling, 5: CopySpelling
TRIAL_COUNT= 1;
TRIAL_YESNO= 2;
TRIAL_CALIBRATION= 3;
TRIAL_FREESPELLING= 4;
TRIAL_COPYSPELLING= 5;

for jj= conditionsOrder,
  tag= condition_tags{jj};
  speller= ['RSVP_' tag];
  i= min(find(isstrprop(tag,'digit')));
  color_mode= tag(1:i-1);
  speed_mode= tag(i:end);

%% Practice 
  msg= sprintf('Press <RETURN> to start %s practice', tag);
  stimutil_waitForInput('msg_next', msg);
  setup_RSVP_feedback;
  pyff('setint', 'trial_type',TRIAL_CALIBRATION);
  pyff('set', 'words',{phrase_practice});
  pyff('setdir','');
  pause(.01)
  fprintf('Ok, starting...\n');
  pyff('play');
  pause(5)
  stimutil_waitForMarker({'S255', 'R  2', 'R  4', 'R  8'});
  pyff('quit');

%% Calibration
  msg= sprintf('Press <RETURN> to start %s RECORDING', tag);
  stimutil_waitForInput('msg_next', msg);
  setup_RSVP_feedback
  pyff('setint','trial_type',TRIAL_CALIBRATION);
  pyff('set', 'words',{phrase_calibration});
  pyff('set', 'present_word_time',2);
  pyff('setdir', 'basename',['calibration_' speller]);
  fprintf('Ok, starting...\n');
  pyff('save_settings', ['calibration_' speller]);
  pyff('play');
  pause(5)
  stimutil_waitForMarker({'S255', 'R  2', 'R  4', 'R  8'});
  pyff('quit');
  bvr_sendcommand('stoprecording');


%% Training the classifier
  bbci= bbci_default;
  bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller, VP_CODE);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller, VP_CODE);
  bbci_bet_prepare
  soa= str2double(speed_mode(1:end-2));
  bbci.setup_opts.ref_ival= [-soa 0];
  bbci.setup_opts.nr_sequences= fbint.sequences_per_trial;
  bbci.setup_opts.nClasses= 30;
  bbci.setup_opts.reject_artifacts=0;
  bbci_bet_analyze
  fprintf('Type dbcont to continue\n');
  keyboard
% you may change bbci.setup_opts and rerun bbci_bet_analyze
% and when satisfied:
  bbci_bet_finish
  close all

%% Copy-spelling
  msg= sprintf('Press <RETURN> to start %s Copy-Spelling', tag);
  stimutil_waitForInput('msg_next', msg);
  if jj==1,
    phrase_copyspelling=phrase_copyspelling1;
  else if jj==2,
     phrase_copyspelling=phrase_copyspelling2;
    else 
      phrase_copyspelling=phrase_copyspelling3;
        end
  end
  setup_RSVP_feedback
  pyff('setint', 'trial_type',TRIAL_COPYSPELLING); 
  pyff('set', 'words',{phrase_copyspelling});
  pyff('set', 'present_word_time',0);
  pyff('setdir', 'basename',['copy_' speller]);
  fprintf('Ok, starting...\n');
  pyff('save_settings', ['copy_' speller]);
  pyff('play');
  pause(5)
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','RSVP_Speller', 'bbci.fb_port', 12345);
  pyff('quit');

%% Free-spelling
  msg= sprintf('Press <RETURN> to start %s Free-Spelling', tag);
  stimutil_waitForInput('msg_next', msg);

  setup_RSVP_feedback
  pyff('setint', 'trial_type',TRIAL_FREESPELLING);
  pyff('set','words','');
  pyff('setdir', 'basename',['free_' speller]);
  pyff('set','show_alphabet',1);
  pyff('set','show_trial_countdown',1);
  fprintf('Ok, starting...\n');
  pyff('save_settings', ['free_' speller]);
  pyff('play');
  pause(1)
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','RSVP_Speller', 'bbci.fb_port', 12345);
  %% To stop the recording: press a button that generates R2 or R4
  %% or type 'ppTrigger(2)' in a second Matlab
  pyff('quit');
  bvr_sendcommand('stoprecording');

  if jj<3,
    fprintf('Press <RETURN> to start oddball measurement.\n');
    pause
    setup_oddball
    pyff('setdir','basename','oddball');
    pause(.01)
    fprintf('Ok, starting...\n'),close all
    pyff('play');
    pause(5)
    stimutil_waitForMarker('stopmarkers','S253');
    pyff('quit');
    bvr_sendcommand('stoprecording');
    % Standard Oddball experiment 
  end
end
fprintf('Experiment finished!\n');

if ~strcmp(VP_CODE, 'Temp');
  save(VP_COUNTER_FILE, 'VP_NUMBER');
end
