%% Feedback settings for the CenterSpeller spellers 
fb = struct();

%% VisualSpellerVE settings:
%fb.log_filename = log_filename;
fb.screenPos = int16(VP_SCREEN);

fb.letterbox_size = [60 60];
fb.osc_size = int16(40);
fb.font_size_phrase = int16(60);
fb.font_size_current_letter = int16(80);
fb.font_size_countdown = int16(150);
% 
% colors:
fb.bg_color = [0 0 0];
fb.countdown_color = 0.6*[1 1 1];
fb.osc_color = [1 1 1];
fb.letter_color = [1 1 1];
fb.phrase_color= 0.6*[1 1 1];
fb.current_letter_color= [1 1 1];

% fb.letter_set = [['A','B','C','D','E'], \
%                    ['F','G','H','I','J'], \
%                    ['K','L','M','N','O'], \
%                    ['P','Q','R','S','T'], \
%                    ['U','V','W','X','Y'], \
%                    ['Z','_','.',',','<']]

fb.fullscreen = int16(0);
fb.use_oscillator = int16(0);
fb.offline= int16(offline_mode);
fb.copy_spelling = int16(1);
fb.debug = int16(0);
fb.nCountdown = int16(3);
fb.nr_sequences = int16(5);
fb.min_dist = int16(2);
tframe = 0.016;    % time for one frame
fb.animation_time = 1-tframe *.5;
fb.stimulus_duration = .1 - tframe *.5;
fb.interstimulus_duration = .1 - tframe *.5;
fb.wait_before_classify = 1;
fb.feedback_duration = 1;
fb.feedback_ErrP_duration = 1;
% 
% % Eyetracker settings
fb.use_eyetracker = int16(0);
fb.et_currentxy = [0, 0];
fb.et_duration = 100;
fb.et_range = 100;
fb.et_range_time = 200;

%fb.use_ErrP_detection = int16(do_ErrP_detection); % ErrP-detection on/off

fb.use_ErrP_detection = int16(0);
%% CenterSpellerVE settings:
fb.letter_radius = int16(40);
fb.speller_radius = int16(250);
fb.font_size_level1 = int16(45);

fb.feedbackbox_size = 200;
fb.fixationpoint_size = 4;
fb.font_size_feedback_ErrP = int16(300);
% 
% stimulus types:
fb.stimulustype_color = int16(conditions(jj,1)); %shapes with color
fb.shape_on = int16(1);
fb.font_size_level2 = int16(130);



% feedback type:
fb.feedback_show_shape = int16(1);
fb.feedback_show_shape_at_center = int16(1);

% level 2 appearance:
fb.level_2_symbols = int16(conditions(jj,2));
fb.level_2_letter_colors = int16(conditions(jj,3));
fb.level_2_animation = int16(1);

% 
% colors:
fb.shape_color = [0.5 0.5 0.5];
fb.stimuli_colors = {[1.0 0.0 0.0], ...
                     [0.0 0.8 0.0], ...
                     [0.0 0.0 1.0], ...
                     [1.0 1.0 0.0], ...
                     [1.0 0.0 0.7], ...
                     [0.9 0.9 0.9]};
fb.letter_color = [0.5, 0.5, 0.5];
fb.feedback_color = [.9, .9, .9];
fb.fixationpoint_color = [1, 1, 1];
fb.feedback_ErrP_color = [0.7, 0.1, 0.1];

% register possible shapes:

% fb.registered_shapes = {'circle':FilledCircle,
%                           'hexagon':FilledHexagon,
%                           'hourglass':FilledHourglass,
%                           'triangle':FilledTriangle,
%                           'rectangle':Target2D};

% define shapes
% fb.shapes=[['triangle',{'size':200}]];
% fb.shapes = [['triangle',  {'size':200}],
%                ['rectangle', {'size':(180., 50), 'orientation':45.}],
%                ['rectangle', {'size':(180., 40), 'orientation':-45.}],
%                ['triangle',  {'size':200, 'orientation':180.}],
%                ['hourglass', {'size':100}],
%                ['circle',    {'radius':90}]];

%% Init Speller
pyff('init','CenterSpellerVE');
pause(1);

%% Send settings
pyff('set',fb);
pause(0.01);

