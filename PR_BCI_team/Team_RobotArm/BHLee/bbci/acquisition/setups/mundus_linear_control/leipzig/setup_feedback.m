%% Feedback settings for the CenterSpeller spellers 
fb = struct();

%% VisualSpellerVE settings:
fb.log_filename = log_filename;
VP_SCREEN = [0 0 800 600]
fb.screenPos = int16(VP_SCREEN);

fb.letterbox_size = [60 60];
fb.osc_size = int16(40);
fb.font_size_phrase = int16(60);
fb.font_size_current_letter = int16(80);
fb.font_size_countdown = int16(150);
fb.desired_phrase = desired_phrase;

% colors:
fb.bg_color = [0 0 0];
fb.phrase_color = [0.2 0.0 1.0];
fb.current_letter_color = [1 0 0];
fb.countdown_color = [0.2 0 1];
fb.osc_color = [1 1 1];

% fb.letter_set = [['A','B','C','D','E'], \
%                    ['F','G','H','I','J'], \
%                    ['K','L','M','N','O'], \
%                    ['P','Q','R','S','T'], \
%                    ['U','V','W','X','Y'], \
%                    ['Z','_','.',',','<']]

fb.fullscreen = int16(0);
fb.use_oscillator = int16(0);
fb.offline= int16(offline_mode);
fb.copy_spelling = int16(0);
fb.debug = int16(0);
fb.nCountdown = int16(3);
fb.nr_sequences = int16(nr_sequences);
fb.min_dist = int16(2);
tframe = 0.016;    % time for one frame
fb.animation_time = 1-tframe *.5;
fb.stimulus_duration = .1 - tframe *.5;
fb.interstimulus_duration = .1 - tframe *.5;
fb.wait_before_classify = 1;
fb.feedback_duration = 1;
fb.feedback_ErrP_duration = 1;

% Eyetracker settings
fb.use_eyetracker = int16(0);
fb.et_currentxy = [0, 0];
fb.et_duration = 100;
fb.et_range = 100;
fb.et_range_time = 200;

%fb.use_ErrP_detection = int16(do_ErrP_detection); % ErrP-detection on/off
fb.use_ErrP_detection = int16(0); % jpg always off


%% CenterSpellerVE settings:
fb.letter_radius = int16(40);
fb.speller_radius = int16(250);
fb.font_size_level1 = int16(45);
fb.font_size_level2 = int16(130);
fb.feedbackbox_size = 200;
fb.fixationpoint_size = 4;
fb.font_size_feedback_ErrP = int16(300);

% stimulus types:
fb.stimulustype_color = int16(1);

% feedback type:
fb.feedback_show_shape = int16(1);
fb.feedback_show_shape_at_center = int16(1);

% colors:
fb.shape_color = [0 0 0];
fb.stimuli_colors = {[1.0 0.0 0.0], ...
                     [0.0 0.8 0.0], ...
                     [0.0 0.0 1.0], ...
                     [1.0 1.0 0.0], ...
                     [1.0 0.0 0.7], ...
                     [0.9 0.9 0.9]};
fb.letter_color = [.6, .6, .6];
fb.feedback_color = [.9, .9, .9];
fb.fixationpoint_color = [1, 1, 1];
fb.feedback_ErrP_color = [0.7, 0.1, 0.1];

%% Init Speller
pyff('init','CenterSpellerVE');

%% Send settings
pyff('set',fb);

