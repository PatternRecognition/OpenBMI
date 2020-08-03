%% Pyff starten
% pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',1);
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks']);

%% Visual Speller
condition_tags= {'ColorAndMotion','Motion','Color'};
order= perms(1:length(condition_tags));
conditionsOrder= order(1+mod(VP_NUMBER-1, size(order,1)),:);

phrase_practice= 'BCI';
phrase_calibration= 'BRAIN_COMPUTER_INTERFACE';
phrase_copyspelling= 'LET_YOUR_BRAIN_TALK.';

for jj= conditionsOrder,
  speller_name= ['CenterDotsCakeSpellerMVEP_' condition_tags{jj}];
  
%% Practice
  fprintf('Press <RETURN> to start %s practice.\n',condition_tags{jj}), pause;
  setup_speller
  % practice
  pyff('setdir','');
  pyff('set','desired_phrase',phrase_practice)
  pyff('setint', 'offline',1);
  pyff('setint', 'feedback_mode',jj - 1);
  pyff('play');
  stimutil_waitForMarker({'S246', 'S255', 'R  2'},'verbose',1);
  pyff('quit');
  
%% Calibration
  fprintf('Press <RETURN> to start %s calibration.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'desired_phrase',phrase_calibration)
  pyff('setint', 'offline',1);
  pyff('setdir', 'basename',['calibration_' speller_name]);
  pyff('setint', 'feedback_mode',jj - 1);
  pyff('save_settings', [TODAY_DIR speller_name]);
  pyff('play');
  stimutil_waitForMarker({'S246', 'S255', 'R  2'});
  pyff('quit');    
  
%% Train the classifier
  bbci= bbci_default;
  bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, '_', VP_CODE);
  bbci_bet_prepare
  fb_opt= pyff_loadSettings([TODAY_DIR speller_name]);
  bbci.setup_opts.nr_sequences= fb_opt.nr_sequences;
  bbci_bet_analyze
%  fprintf('Type ''dbcont'' to continue\n');
%  keyboard
  bbci_bet_finish
  close all
  
%% Freespelling
  fprintf('Press <RETURN> to start %s free-spelling.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'desired_phrase','');
  pyff('set','copy_spelling',int16(0));
  pyff('setint', 'offline',0);
  pyff('setint', 'feedback_mode',jj - 1);
  pyff('setdir', 'basename',['free_' speller_name]);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Free-spelling run finished.\n')
  pyff('quit');
    
%% Online copy-spelling
  fprintf('Press <RETURN> to start %s copy-spelling.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'desired_phrase',phrase_copyspelling);
  pyff('setint', 'offline',0);
  pyff('setint', 'feedback_mode',jj - 1);
  pyff('setdir', 'basename',['copy_' speller_name]);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
  pyff('quit');
%% end of the loop
end

if ~strcmp(VP_CODE, 'Temp');
  save(VP_COUNTER_FILE, 'VP_NUMBER');
end
