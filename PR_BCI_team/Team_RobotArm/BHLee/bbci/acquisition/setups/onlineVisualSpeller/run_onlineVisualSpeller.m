%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n'), pause

%% Vormessungen
% Run artifact measurement, oddball practice and oddball test
% vormessungen

%% Pyff starten
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0);
%% Basic settings
nVP = 14;   % number of vp
% fprintf('Number of VP is %d, is this right??\n',nVP), pause
spellers = {'CenterSpellerVE' 'HexoSpellerVE' 'CakeSpellerVE'};
calib_prefix = 'calibration_';
online_copy_prefix = 'online_copy_';     % copy spelling
online_free_prefix = 'online_free_';     % free spelling
copy_phrase = 'LET_YOUR_BRAIN_TALK';
free_phrase = '';
% Counterbalance order of the spellers
conditions = {[1 2 3] [1 3 2] [2 1 3] [2 3 1] [3 1 2] [3 2 1]};
currentCondition = conditions{mod(nVP-1,6)+1};

% for iii=1:numel(spellers)
iii=3

currentSpeller = spellers{currentCondition(iii)};
% fprintf('Press <RETURN> to proceed with %s\n',currentSpeller),pause

%% Calibration
offline_mode = 1;
run_calibration

%% Train the classifier
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, calib_prefix, currentSpeller, '_', VP_CODE);
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', currentSpeller, '_', VP_CODE);
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type dbcont to continue\n');
keyboard
bbci_bet_finish
close all

%% Copy-spelling Experiment
%startup_pyff
offline_mode = 0;
fprintf('Press <RETURN> to start %s copy-spelling experiment.\n',currentSpeller), pause;
setup_speller
% pyff('set','desired_phrase',copy_phrase);
pyff('set','desired_phrase','');
pyff('setdir','basename',[online_copy_prefix currentSpeller '_']);
fprintf('Ok, starting...\n');
pyff('play');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
% stimutil_waitForMarker({'R  1', ['S' num2str(COPYSPELLING_FINISHED)]});

fprintf('Copy-spelling run finished.\n')
pyff('stop');
pyff('quit');

%% Free spelling experiment
%startup_pyff
offline_mode = 0;
% free_phrase = input('Enter phrase (CAPITAL LETTERS):','s');
fprintf('\nThe chosen phrase is ''%s''.\n',free_phrase)
fprintf('Press <RETURN> to start %s free-spelling experiment.\n',currentSpeller), pause;
setup_speller
% pyff('set','desired_phrase',free_phrase);
pyff('setdir','basename',[online_copy_prefix currentSpeller '_']);
pause(4)
fprintf('Ok, starting...\n');

pyff('play');
% bbci_bet_apply(bbci_setup, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
%  'phrase','stop');

fprintf('Free-spelling run finished.\n')
pyff('stop');
pyff('quit');
fprintf('Finished all %s runs.\n',currentSpeller),pause(4)
%%
end
%%
fprintf('Finshed experiment.\n');
