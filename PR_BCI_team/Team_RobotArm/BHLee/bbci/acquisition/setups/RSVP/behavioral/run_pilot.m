%% Basic parameters
filename = 'RSVP_';
% minTargets = 0;       % Minimum number of targets for pre/post sequence
% maxTargets = 2;       % Maximum number of targets for pre/post sequence
% custom_pre_sequences = {[minTargets maxTargets]};
% custom_post_sequences = {[minTargets maxTargets]};
% practice_pre_sequences = {};
% practice_post_sequences = {};

% Reset random-generator seed to produce new random numbers
% Take cputime in ms as basis
rand('seed',cputime*1000)

%% ** Startup pyff **
close all
pyff('startup','a','D:\svn\bbci\python\pyff\src\Feedbacks');
% pyff('startup','a','D:\svn\bbci\python\pyff\src\Feedbacks', 'gui',1);
bvr_sendcommand('viewsignals');
pause(4)
%% Experimental settings
alternating_colors = [0 1];   % Color off, Color on
alternating_colors_name = {'NoColor','Color'};
stim_durations_name = {'83ms','100ms','116ms','133ms'};
stim_durations = [.070 0.09 0.103 0.119];  % in s 
% stim_durations = [3 3 3 3];  % in s 

test_words =  {'wiNkT','fJORd','LEXUS' ... 
          'SpHiNX','qUARz','vOdkA' ...
          'yAcHT','GEBOT'};
        
rand_word=randperm(length(test_words));

%Experimental subconditions: [nocolor-color stim_duration]
conditions = {[1 1] [1 2] [1 3] [1 4] [2 1] [2 2] [2 3] [2 4]};

% Order of conditions
rand_idx = randperm(length(conditions));

% condOrder = {[1 2 3] [1 3 2] [2 1 3] [2 3 1] [3 1 2] [3 2 1]};
% conditions = conditions(condOrder{1+mod(vp_number,6)});

%Trial type in RSVP speller
% 1: Count, 2: YesNo, 3: Calibration, 4: FreeSpelling, 5: CopySpelling
TRIAL_COUNT = 1;
TRIAL_YESNO = 2;
TRIAL_CALIBRATION = 3;
TRIAL_FREESPELLING = 4;
TRIAL_COPYSPELLING = 5;


%% **RUN** Behavioral experiment

fprintf('Press <RETURN> to start the experiment.\n'),pause

for current_idx = 1:length(rand_idx),
cc = conditions{rand_idx(current_idx)};

%% Calibration
fprintf('Press <RETURN> to start the %s %s calibration.\n',alternating_colors_name{cc(1)}, stim_durations_name{cc(2)}), pause 
close all
setup_RSVP_feedback
fbint.trial_type=TRIAL_CALIBRATION; %1: Count, 2: YesNo, 3: Calibration, 4: FreeSpelling, 5: CopySpelling
pyff('setint','trial_type',fbint.trial_type); 
fb.words= {test_words{rand_word(current_idx)}};
pyff('set','words',fb.words);
fb.present_word_time =2;
pyff('set','present_word_time',fb.present_word_time);
% log_filename = [TODAY_DIR VP_CODE '.log'];
pyff('setdir','basename',[filename 'behav_' alternating_colors_name{cc(1)} '_' stim_durations_name{cc(2)}]);
% pyff('setdir','');
fprintf('Ok, starting...\n'),close all
pyff('play');
pause(5)
% stimutil_waitForMarker('stopmarkers','S253');
fprintf('%s %s calibration finished? If yes, press <RETURN>\n',  alternating_colors_name{cc(1)}, stim_durations_name{cc(2)}),pause
pause(4);
pyff('stop');
pyff('quit');
bvr_sendcommand('stoprecording');

%%
end

fprintf('Experiment finished!\n');