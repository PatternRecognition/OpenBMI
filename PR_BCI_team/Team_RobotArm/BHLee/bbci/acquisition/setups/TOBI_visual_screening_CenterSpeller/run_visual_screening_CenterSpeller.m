%% Run Oddball Practise and Oddball Test
run_oddball;


%% Center Speller
% Testing three (four) conditions
% 1) Shapes with different color, level 2 Shapes diffenrent color
% 2) Shapes with different color, level 2 No shapes, letters in different
% color
% 3) Shapes with same color, level 2 Shapes with same color
% ( 4) Same shape with different color (level 1 & 2)  )
%initialize variables
% COPYSPELLING_FINISHED = 246;
% RUN_END = 253;

global RESTART_FEEDBACK

shapes_color_on=[int16(0) int16(1)];
level_2_symbols_on=[int16(0) int16(1)];
letter_color_on=[int16(0) int16(1)];

condition_tags= {'', 'ColorOnlyAndLetters', 'ShapeOnly'}; %using only ''
conditions=[shapes_color_on(2) level_2_symbols_on(2) letter_color_on(1);
    shapes_color_on(2) level_2_symbols_on(1) letter_color_on(2);
    shapes_color_on(1) level_2_symbols_on(2) letter_color_on(1)];
order={[1 2 3] [1 3 2] [2 1 3] [2 3 1] [3 1 2] [3 2 1]};
conditionsOrder= order{1+mod(VP_NUMBER-1, length(order))};


phrase_practice= 'BCI';
% phrase_calibration= 'BRAIN_COMPUTER_INTERFACE';
phrase_calibration= 'KREUZNACHER_DIAKONIE';
phrase_copyspelling= 'MIT_GEDANKEN_SCHREIBEN';

% for testing
%   phrase_calibration= 'ABC';
%   phrase_copyspelling= 'XYZ';

speller_name= 'CenterSpeller';

%% Calibration
offline_mode=int16(1);
fprintf('Press <RETURN> to start %s practice.\n',speller_name), pause;
setup_speller
% practice
pyff('set','desired_phrase',phrase_practice)
pyff('setdir','');
pyff('play'); pause(1);
stimutil_waitForMarker({'S246', 'S253', 'S255', 'R  2', 'R  4', 'R  8'});
pyff('stop'); pyff('quit');

% recording
fprintf('Press <RETURN> to start %s calibrations.\n',speller_name), pause;

jj = 1; %-->  condition ''
setup_speller
pyff('set','desired_phrase',phrase_calibration)
pyff('setdir','basename',['calibration_' speller_name]);
pyff('play'); pause(2);
stimutil_waitForMarker({'S246', 'S253', 'R  2', 'R  4', 'R  8'});
pyff('stop'); pyff('quit');

%% Train the classifier
bbci= bbci_default;
bbci.train_file= strcat(TODAY_DIR, 'calibration_', speller_name, VP_CODE);
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller_name, '_', VP_CODE);
bbci_bet_prepare
bbci.setup_opts.nr_sequences= fb.nr_sequences;
bbci_bet_analyze
fprintf('Type ''dbcont'' to continue\n');
keyboard
bbci_bet_finish
close all

%% Online copy-spelling
offline_mode=int16(0);
fprintf('Press <RETURN> to start %s copy-spelling experiment.\n',condition_tags{jj}), pause;
setup_speller
pyff('set','desired_phrase',phrase_copyspelling);
pyff('setdir','basename',['copy_' speller_name]);
pyff('play');
RESTART_FEEDBACK= 1; pause(2);
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
fprintf('Copy-spelling run finished.\n')
pyff('stop'); pyff('quit');

%% freespelling
fprintf('Press <RETURN> to start %s free-spelling experiment.\n',condition_tags{jj}), pause;
setup_speller
pyff('set','desired_phrase','');
pyff('set','copy_spelling',int16(0));
pyff('setdir','basename',['free_' speller_name]);
pyff('play'); pause(2);
RESTART_FEEDBACK= 1;
bbci_bet_apply(bbci.save_name, 'bbci.feedback','ERP_Speller', 'bbci.fb_port', 12345);
fprintf('Copy-spelling run finished.\n')
pyff('stop'); pyff('quit');


if ~strcmp(VP_CODE, 'Temp');
    save(VP_COUNTER_FILE, 'VP_NUMBER');
end

