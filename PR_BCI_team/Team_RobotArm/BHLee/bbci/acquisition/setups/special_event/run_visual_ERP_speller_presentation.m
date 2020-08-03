%% Start pyff
pyff('startup', 'a', [BCI_DIR 'python/pyff/src/Feedbacks/_VisualSpeller'], 'gui', 0);

spellers = {'HexoSpeller', 'CenterSpeller'};

%%
% speller = spellers{1}
for speller = spellers,

%% Setup

phrase_calibration = 'BRAIN_COMPUTER_INTERFACE';
phrase_practice = 'BCI';
phrase_online = '';%LET_YOUR_BRAIN_TALK';
copyspelling = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Speller Calibration %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
basename = ['calibration_' speller '_'];
offline_mode = 1;
nr_sequences = 5;

%% Calibration **practice**
fprintf('Press <RETURN> to start speller calibration PRACTICE.\n'); pause;
log_filename = [TODAY_DIR basename 'practice_' VP_CODE '.log']; %#ok<*NASGU>
desired_phrase = phrase_practice;

setup_visual_ERP_speller_presentation_spellerSetup
pyff('setdir','');
pyff('play');
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)]});
pyff('stop');
pyff('quit');


%% Calibration **RUN**
fprintf('Press <RETURN> to start speller calibration.\n'); pause;
log_filename = [TODAY_DIR basename VP_CODE '.log'];
desired_phrase = phrase_clibration;

setup_visual_ERP_speller_presentation_spellerSetup 
pyff('setdir', 'basename', basename);
pyff('play');
stimutil_waitForMarker({['S' num2str(COPYSPELLING_FINISHED)]});
pyff('stop');
pyff('quit');
fprintf('Speller calibration finished.\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train the classifiers %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, basename, VP_CODE);
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller, '_', VP_CODE);
bbci_bet_prepare
bbci_bet_analyze
fprintf('Type dbcont to continue\n');
keyboard
bbci_bet_finish
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Online Spelling %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
basename = ['online_' speller '_'];
nr_sequences = 5;
offline_mode = 0;

%% Start Online Fixed Spelling:
fprintf('Press <RETURN> to start next block.\n'); pause;
log_filename = [TODAY_DIR basename VP_CODE '.log'];

desired_phrase = phrase_online;
setup_visual_ERP_speller_presentation_spellerSetup
pyff('setdir', 'basename', basename);
pyff('play'); pause(1)
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
pyff('stop');
pyff('quit');

%%
end