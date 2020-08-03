%% Pyff starten
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks']);

speller_list= {'HexoSpeller', 'CenterSpeller'};

phrase_practice= 'BCI';
phrase_calibration= 'LET_YOUR_BRAIN_TALK';
phrase_copyspelling= 'X:ENIUS';

for jj= 1:length(speller_list),
  speller_name= speller_list{jj};
  
  %% Practice
  fprintf('Press <RETURN> to start %s practice.\n',speller_name), pause;
  setup_speller
  % practice
  pyff('setdir','');
  pyff('set','desired_phrase',phrase_practice)
  pyff('setint', 'offline',1);
  pyff('play');
  stimutil_waitForMarker({'S255', 'S246', 'R  2'},'verbose',1);
  pyff('quit');
  
  % Calibration
  fprintf('Press <RETURN> to start %s calibration.\n',speller_name), pause;
  setup_speller
  pyff('set', 'phrase',phrase_calibration)
  pyff('setint', 'offline',1);
  pyff('setint', 'nr_sequences',7);
  pyff('setdir', 'basename',['calibration_' speller_name]);
  pyff('save_settings', speller_name);
  pyff('play');
  stimutil_waitForMarker({'S255', 'S246', 'R  2'},'verbose',1);
  pyff('quit');    
  
  %% Train the classifier

  bbci= bbci_default;
  bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, '_', VP_CODE);
  
    bbci.save_name= 'D:\data\bbciRaw\VPpn_11_03_17\bbci_classifier_CenterSpeller_VPpn';
  
  bbci_bet_prepare
%  feedback_settings= pyff_loadSettings([TODAY_DIR speller_name]);
%  bbci.setup_opts.nr_sequences= feedback_settings.nr_sequences;
  bbci.setup_opts.nr_sequences= 5;
  bbci_bet_analyze
  fprintf('Type ''dbcont'' to continue\n');
  keyboard
  bbci_bet_finish
  close all
    
  %% Online copy-spelling
  fprintf('Press <RETURN> to start %s copy-spelling.\n',speller_name), pause;
  setup_speller
  
  fb.classes =  {'left', 'right'}
  fb.classesDirections =  {'left', 'right'}
  pyff('set',fb); 
  pyff('set', 'desired_phrase',phrase_copyspelling);
  pyff('setint', 'offline',0);
  pyff('setdir', 'basename',['copy_' speller_name]);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
  pyff('quit');
  
  %% Freespelling
  fprintf('Press <RETURN> to start %s free-spelling.\n',speller_name), pause;
  setup_speller
  pyff('set', 'desired_phrase','');
  pyff('setint', 'offline',0);
  pyff('setdir', 'basename',['free_' speller_name]);
  pyff('play');
  bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Free-spelling run finished.\n')
  pyff('quit');
end
