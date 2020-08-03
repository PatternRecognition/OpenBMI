pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0);

condition_tags= {'Upsize', 'Intensify', 'Rotate'};
order= perms(1:length(condition_tags));
conditionsOrder= order(1+mod(VP_NUMBER-1, size(order,1)),:);

TRIAL_CALIBRATION= 1;
TRIAL_COPYSPELLING= 3;
TRIAL_FREESPELLING= 2;

phrase_practice= {'BCI'};
phrase_calibration= {'BRAIN_COMPUTER_INTERFACE'};
phrase_copyspelling= {'LET_YOUR_BRAIN_TALK.'};

for jj= conditionsOrder,
  
  speller_name= ['MatrixSpeller' condition_tags{jj}];
  
  %% Practice
  fprintf('Press <RETURN> to start %s practice.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set','phrases',phrase_practice)
  pyff('setint', 'trial_type',TRIAL_CALIBRATION);
  pyff('setdir','');
  pyff('setint', 'stimulus_type',jj);
  pyff('play');
  stimutil_waitForMarker({'S255', 'R  2', 'R  4', 'R  8'});
  pyff('quit');
  
  %% Calibration
  fprintf('Press <RETURN> to start %s calibration.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'phrases',phrase_calibration)
  pyff('setint', 'trial_type',TRIAL_CALIBRATION);
  pyff('setdir', 'basename',['calibration_' speller_name]);
  pyff('save_settings', speller_name);
  pyff('setint', 'stimulus_type',jj);
  pyff('play');
  stimutil_waitForMarker({'S247', 'S255', 'R  2', 'R  4', 'R  8'});
  pyff('quit');    
  
  %% Train the classifier
  bbci= bbci_default;
  bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, '_', VP_CODE);
  bbci_bet_prepare
  feedback_settings= pyff_loadSettings('project2010_biomed02/MatrixSpeller_default');
  bbci.setup_opts.nr_sequences= feedback_settings.nr_sequences;
  bbci.setup_opts.matrix_columns= feedback_settings.matrix_columns;
  bbci.setup_opts.nr_symbols= length(feedback_settings.symbols);
  bbci_bet_analyze
  fprintf('Type ''dbcont'' to continue\n');
  keyboard
  bbci_bet_finish
  close all
    
  %% Online copy-spelling
  fprintf('Press <RETURN> to start %s copy-spelling.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'phrases',phrase_copyspelling);
  pyff('setint', 'trial_type',TRIAL_COPYSPELLING);
  pyff('setdir', 'basename',['copy_' speller_name]);
  pyff('setint', 'stimulus_type',jj);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_MatrixSpeller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
  pyff('quit');
  
  %% Freespelling
  fprintf('Press <RETURN> to start %s free-spelling.\n',condition_tags{jj}), pause;
  setup_speller
  pyff('set', 'desired_phrase','');
  pyff('setint', 'trial_type',TRIAL_FREESPELLING);
  pyff('setdir', 'basename',['free_' speller_name]);
  pyff('setint', 'stimulus_type',jj);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_MatrixSpeller', 'bbci.fb_port', 12345);
  fprintf('Free-spelling run finished.\n')
  pyff('quit');
end

if ~strcmp(VP_CODE, 'Temp');
  save(VP_COUNTER_FILE, 'VP_NUMBER');
end
