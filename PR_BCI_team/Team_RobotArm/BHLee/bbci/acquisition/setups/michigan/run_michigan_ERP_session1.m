%% Impedanzcheck
bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n'), pause
bvr_sendcommand('viewsignals');

%% Vormessungen
% Run artifact measurement, oddball practice and oddball test
% vormessungen

%% Basic settings
speller_name= 'HexoSpeller';
online_copy_prefix = 'online_copy_';     % copy spelling
online_free_prefix = 'online_free_';     % free spelling
phrase_practice= 'BCI';
phrase_calibration= 'LET_YOUR_BRAIN_TALK.';
phrase_copyspelling= 'BRAIN_COMPUTER_INTERFACE';

%% Pyff starten
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'bvplugin',0,'gui',0);
% send_xmlcmd_udp('init', '127.0.0.1', 12345); % not sure whether this is required
acqFolder = [BCI_DIR 'acquisition/setups/michigan/'];
SPELLER_file = [acqFolder speller_name];

%% Practice
stimutil_waitForInput('msg_next', ['to start ' speller_name ' practice']);
setup_ERP_speller
pyff('set','desired_phrase',phrase_practice)
pyff('setint', 'offline',1);
pyff('play');
stimutil_waitForMarker(RUN_END,'verbose',1);
pyff('quit');
 
%% Calibration
stimutil_waitForInput('msg_next', ['to start ' speller_name ' calibration']);
setup_ERP_speller
pyff('set', 'desired_phrase',phrase_calibration)
pyff('setint', 'offline',1);
pyff('save_settings', speller_name);
pyff('play', 'basename', ['calibration_' speller_name], 'impedances', 0);
stimutil_waitForMarker(RUN_END);
pyff('quit');    

%% Train the classifier
bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, VP_CODE);
% to remove these 2 lines or check that the channels exist
bbci.setup_opts.clab_erp= 'Pz';
bbci.setup_opts.clab_rsq= {'Cz','Pz'};
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type dbcont to continue\n');
keyboard
bbci_bet_finish
close all
bbci= bbci_apply_loadSettings(bbci.save_name);
%bbci= bbci_apply_cleanup(bbci);
bbci.source.acquire_fcn= @bbci_acquire_bv;
bbci.source.acquire_param = {struct('fs',100)};
bbci.source.marker_mapping_fcn= '';
bbci.feature.param{2}{1}= bbci.feature.param{2}{1} + bbci.control.condition.overrun;
bbci.control.fcn= @bbci_control_ERP_Speller_binary;
bbci.control.param= {struct('nClasses',6, 'nSequences',NR_SEQUENCES)};
%bbci.control.condition.marker= [11:16,21:26,31:36,41:46];
%bbci.feedback.receiver= 'pyff';
bbci.log.output= 'screen';
bbci.quit_condition.marker= [4 5 RUN_END];
bbci.quit_condition.running_time= 60*10;    %% quit after 10 minutes

%% Copy-spelling Experiment
%startup_pyff
stimutil_waitForInput('msg_next', ['to start ' speller_name ' copy-spelling']);
setup_ERP_speller
pyff('setint','offline',0);
pyff('set','desired_phrase',phrase_copyspelling);
pyff('play','basename',['copy_' speller_name], 'impedances', 0);
bbci_apply(bbci);
fprintf('Copy-spelling run finished.\n')
pyff('quit');

%% Free spelling experiment
fprintf('Send marker %d from a second Matlab to stop free spelling.\n', RUN_END);
stimutil_waitForInput('msg_next', ['to start ' speller_name ' free-spelling']);
setup_ERP_speller
pyff('setint','offline',0);
pyff('set','desired_phrase','');
pyff('play','basename',['free_' speller_name], 'impedances', 0);
bbci_apply(bbci)
fprintf('Free-spelling run finished.\n')
pyff('quit');
