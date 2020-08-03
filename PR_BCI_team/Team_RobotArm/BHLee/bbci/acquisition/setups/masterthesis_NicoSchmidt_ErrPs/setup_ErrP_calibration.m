

%% Feedback settings for the three spellers 
fb= struct();

fb.offline= int16(0);
fb.debug= int16(0);
fb.screenPos = int16(VP_SCREEN);

fb.error_rate = 0.2;
fb.error_mindist= int16(2);
fb.len_error_sequence= int16(100);
fb.preceding_nonerrors= int16(4);
fb.arrow_color = [1. 1. 1.];
fb.arrow_size = [30. 10.];
fb.arrow_round_time = 1.5;
fb.arrow_precision= int16(10);
fb.arrow_start_orientation= int16(-90);
fb.stop_after_response= int16(1);
fb.stop_after_rounds= int16(2);
fb.arrow_random_start= int16(1);
fb.arrow_reset_after_trial= int16(0);
fb.one_level_only = int16(0);
% fb.one_level_letters = {'A','E','I','D','N','<'};
fb.chooe_second_at_error = int16(1);

fb.nCountdown = int16(5);
fb.animation_time = 0.5;
fb.wait_before_classify = 1.;
fb.feedback_show_shape = int16(1);
fb.feedback_show_shape_at_center = int16(1);
fb.desired_phrase = '';


% Visual settings
fb.screenPos =  int16(VP_SCREEN);
fb.fullscreen = int16(1);
fb.stimuli_colors = {[1.0 0.0 0.0], ...
                     [0.0 0.8 0.0], ...
                     [0.0 0.0 1.0], ...
                     [1.0 1.0 0.0], ...
                     [1.0 0.0 0.7], ...
                     [0.9 0.9 0.9]};
fb.shape_color = [0 0 0];
fb.letter_color = [.6 .6 .6];
fb.fixationpoint_color = [1.0, 0.0, 0.0];

% Eyetracker
fb.use_eyetracker = int16(0);
% fb.et_duration = 100;
% fb.et_range = 200;       % Maximum acceptable distance between target and actual fixation
% fb.et_range_time = 200; 


%% Init Speller
pyff('init','CenterSpellerErrPCalibration');

%% Send settings
pyff('set',fb);

