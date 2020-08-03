%% Feedback settings for the three spellers 
fb= struct();

%% VisualSpellerVE settings:
fb.log_filename = log_filename;
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

fb.fullscreen = int16(1);
fb.use_oscillator = int16(0);
fb.offline= int16(0);
fb.copy_spelling = int16(0);
fb.debug = int16(0);
fb.nCountdown = int16(5);
fb.nr_sequences = 1;
fb.min_dist = int16(2);
tframe = 0.016;    % time for one frame
fb.animation_time = 0.5-tframe *.5;
fb.stimulus_duration = .1 - tframe *.5;
fb.interstimulus_duration = .1 - tframe *.5;
fb.wait_before_classify = 1.;
fb.feedback_duration = 1.;
fb.feedback_ErrP_duration = 1;

% Eyetracker settings
fb.use_eyetracker = int16(0);
fb.et_currentxy = [0, 0];
fb.et_duration = 100;
fb.et_range = 100;
fb.et_range_time = 200;

fb.use_ErrP_detection = int16(0); % ErrP-detection on/off


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
fb.feedback_color = [0.9, 0.9, 0.9];
fb.fixationpoint_color = [1.0, 1.0, 1.0];
fb.feedback_ErrP_color = [0.7, 0.1, 0.1];

% register possible shapes:
% fb.registered_shapes = {'circle':FilledCircle,
%                           'hexagon':FilledHexagon,
%                           'hourglass':FilledHourglass,
%                           'triangle':FilledTriangle,
%                           'rectangle':Target2D};

% define shapes
% fb.shapes = [['triangle',  {'size':200}],
%                ['rectangle', {'size':(180., 50), 'orientation':45.}],
%                ['rectangle', {'size':(180., 40), 'orientation':-45.}],
%                ['triangle',  {'size':200, 'orientation':180.}],
%                ['hourglass', {'size':100}],
%                ['circle',    {'radius':90}]];


%% CenterSpellerErrPCalibration settings:
fb.wait_between_triggers = int16(50);
fb.error_rate = 0.15;
fb.error_mindist = int16(2);
fb.len_error_sequence = int16(100);
fb.preceding_nonerrors= int16(4);
fb.arrow_color = [0.6 0.6 0.6];
fb.arrow_size = [30 10];
fb.arrow_round_time = 1.5;
fb.arrow_precision = 10;
fb.arrow_start_orientation = -90;
fb.stop_after_response = int16(1);
fb.stop_after_rounds = int16(2);
fb.arrow_random_start = int16(1);
fb.arrow_reset_after_trial = int16(0);
fb.one_level_only = int16(0);
% fb.one_level_letters = ['A','E','I','D','N','<'];
fb.choose_second_at_error = int16(0);


%% Init Speller
pyff('init','CenterSpellerErrPCalibration');

%% Send settings
pyff('set',fb);

