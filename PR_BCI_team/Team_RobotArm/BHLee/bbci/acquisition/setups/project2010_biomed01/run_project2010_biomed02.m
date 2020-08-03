pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0);

condition_tags= {'RandSeq', 'RegSeq'};
order= perms(1:length(condition_tags));
conditionsOrder= order(1+mod(VP_NUMBER-1, size(order,1)),:);

phrase_practice= 'BCI';
phrase_calibration= 'BRAIN_COMPUTER_INTERFACE';
phrase_copyspelling= 'LET_YOUR_BRAIN_TALK.';

for jj= conditionsOrder,
  speller_name= ['CenterSpeller' condition_tags{jj}];
  
  %% Practice
  fprintf('Press <RETURN> to start %s practice.\n',condition_tags{jj}), pause;
  setup_speller
  % practice
  pyff('set','phrase',phrase_practice)
  pyff('setdir','');
  pyff('play');
  stimutil_waitForMarker({['S247' 'S255' 'R  2' 'R  4' 'R  8'});
  pyff('stop'); pyff('quit');
  
  % Calibration
  fprintf('Press <RETURN> to start %s calibration.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'phrase',phrase_calibration)
  pyff('setdir', 'basename',['calibration_' speller_name]);
  pyff('save_settings', speller_name);
  pyff('play');
  stimutil_waitForMarker({['S247' 'S255' 'R  2' 'R  4' 'R  8'});
  pyff('stop'); pyff('quit');    
  
  %% Train the classifier
  bbci= bbci_default;
  bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, '_', VP_CODE);
  bbci_bet_prepare
  bbci.setup_opts.nr_sequences= fbint.nr_sequences;
  bbci_bet_analyze
  fprintf('Type ''dbcont'' to continue\n');
  keyboard
  bbci_bet_finish
  close all

  %% Online copy-spelling
  fprintf('Press <RETURN> to start %s copy-spelling.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'phrase',phrase_copyspelling);
  pyff('setdir', 'basename',['copy_' speller_name]);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
  pyff('stop'); pyff('quit');
  
  %% Freespelling
  fprintf('Press <RETURN> to start %s free-spelling.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'desired_phrase','');
  pyff('setdir', 'basename',['free_' speller_name]);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
  pyff('stop'); pyff('quit');
end

if ~strcmp(VP_CODE, 'Temp');
  save(VP_COUNTER_FILE, 'VP_NUMBER');
end
