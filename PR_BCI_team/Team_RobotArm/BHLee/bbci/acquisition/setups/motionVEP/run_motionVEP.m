%% Impedanzcheck
% bvr_sendcommand('checkimpedances');
% fprintf('Prepare cap. Press <RETURN> when finished.\n'), pause
% 
% %% Startup pyff
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks\_VisualSpeller'],'gui',0);
% bvr_sendcommand('viewsignals'); pause(5)

%% Basic settings
nVP = 17;   % number of vp
fprintf('Number of VP is %d, is this right??\n',nVP), pause
spellers = {'overt_cake' 'covert_cake' 'CenterSpellerMVEP'};
copy_phrase = 'LET_YOUR_BRAIN_TALK';
calib_prefix = 'calibration_';
online_copy_prefix = 'copy_';     % copy spelling
online_free_prefix = 'free_';     % free spelling
COPYSPELLING_FINISHED = 246;
% FINISHED = 246;
RUN_END = 253;

free_phrase = '';

% Counterbalance order of the spellers
conditions = {[1 2 3] [1 3 2] [2 1 3] [2 3 1] [3 1 2] [3 2 1]};
cc = conditions{mod(nVP-1,6)+1};  % cc = current condition

%%

for iii=1:numel(spellers)
%%
cs = spellers{cc(iii)}; % cs = current speller

% Calibration
offline_mode = 1;
run_calibration

%% Train the classifier
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, calib_prefix, cs, '_', VP_CODE);
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', cs, '_', VP_CODE);
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type ''dbcont'' to continue\n');
keyboard
bbci_bet_finish
close all

%% Copy-spelling Experiment
offline_mode = 0;
fprintf('Press <RETURN> to start %s copy-spelling experiment.\n',cs), pause;
setup_speller
pyff('set','copy_spelling',1);
fprintf('Ok, starting...\n');
% pyff('set','desired_phrase',copy_phrase);
pyff('set','desired_phrase','LET_YOUR_BRAIN_TALK');
pyff('setdir','basename',[online_copy_prefix cs '_']);
pyff('play');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
  fprintf('Copy-spelling run finished.\n')
pyff('stop');
bvr_sendcommand('stoprecording');
pyff('quit');

%% Free spelling experiment
offline_mode = 0;
% fprintf('Press <RETURN> to start %s free-spelling experiment.\n',cs), pause;
% free_phrase = input('Enter phrase:','s');
% free_phrase = strrep(upper(free_phrase),' ','_');
free_phrase = '';
% fprintf('\nThe chosen phrase is ''%s''.\n',free_phrase)
fprintf('Press <RETURN> to start %s free-spelling experiment.\n',cs), pause;
setup_speller
fprintf('Ok, starting...\n');
pyff('set','desired_phrase',free_phrase);
pyff('setdir','basename',[online_free_prefix cs '_']), pause(4)
pyff('play');
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
% bbci_bet_apply(bbci_setup, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);

fprintf('Free-spelling run finished.\n')
pyff('stop');
bvr_sendcommand('stoprecording');
pyff('quit');
fprintf('Finished %s runs.\n',cs),pause(4)
%%
end
%%
fprintf('Finshed experiment.\n');
