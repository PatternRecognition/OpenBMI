%% Feedback settings for the CenterSpeller spellers 
fb = struct();
fbint = struct();

%% VisualSpellerVE settings:

fb.log_filename = log_filename;

% Trigger (NIRS 4-bit format compatible -> 1-14)
fbint.CUE = 1;      % 1-4
fbint.TARGET = 5;   % 5=X 6=+
fbint.KEY_1 = 7;
fbint.INVALID_FIXATION = 13;
% RUN_START ist 254 (im NIRS 14)
% RUN_END ist 255 (im NIRS 15)

% Trigger (only EEG compatible -> will always yield 0000=0 in NIRS)
% fbint.TRIAL_START = 31;
% fbint.COUNTDOWN_START = 63;
% fbint.MASKER = 79;
fbint.TRIAL_START = 32;
fbint.COUNTDOWN_START = 48;
fbint.MASKER = 64;
% fbint.FIXATION_START = 95;

% Experimental design
fbint.nr_elements = nDirections; % number of directions
fbint.nr_stimuli = 5;
fb.p_correct = 0.7; % fraction of correct trials (cue=target) / IN 

% Graphics
fbint.geometry = VP_SCREEN;
fbint.fullscreen = 0;
fbint.fixationpoint_radius = 6;
fb.arrow_size = [60 12];
fbint.screen_radius = 400;
fbint.circle_radius = 100;
fbint.target_size = 120;
fbint.countdown_font_size = 150;
fbint.input_box_size = [600, 70];
fbint.input_text_size = 60;
% self.target_symbols = [u'\u00D7', '+']; % first target, second nontarget
fbint.use_masker = 1;

% Timing
fbint.nr_countdown = 5;
fb.fixation_duration = 1;
fb.cue_duration = 0.2;
% fb.stimulus_jitter = [-0.5, 1.5];
fb.stimulus_jitter = {[0, 0],[-.3,.3],[-.3,.3],[-.3,.3],[0,0]};
fb.trial_interval = [1.0, 6.0];
fb.target_duration = 0.3;
fb.masker_duration = 0.14;
fb.end_duration = .8;
fb.pause_duration = [5 15];
fb.idle_fs = 0.05;
fb.idle_max_time = 0.1;

% Colors
fb.arrow_color = [1 1 1]; % masker cues
fb.cue_color = [.5 .5 .5]; % attended cue
fb.bg_color = 'black';
fb.countdown_color = [.8 0 1];
fb.target_color = [.8 .8 .8];
fb.circle_color = [.75 .75 .75];
fb.circle_textcolor = [0 0 0];
fb.fixationpoint_color = [1 1 1];
fb.input_box_color = [.2 .2 .2];
fb.input_text_color = [1 1 1];
% fb.cue_colors = [(0.0, 1.0, 0.8), 
%                    (0.2, 0.0, 0.8), 
%                    (0.4, 1.0, 0.8), 
%                    (0.2, 0.0, 0.8), 
%                    (0.7, 1.0, 0.8), 
%                    (0.2, 0.0, 0.8)]
% fb.cue_bordercolor = (0.0, 0.0, 1.0)
% fb.user_cuecolor = 4; % the index of the color, which is used as cue 

fbint.multi_stimulus_mode = 1;
fbint.start_angle = 0;

% % Eyetracker settings
fbint.use_eyetracker = 0;
fbint.et_range = 150;
fbint.et_range_time = 200;

%% Init Speller
pyff('init','CovertAttentionVE');
pause(1)

%% Send settings
pyff('set',fb);
pyff('setint',fbint);


