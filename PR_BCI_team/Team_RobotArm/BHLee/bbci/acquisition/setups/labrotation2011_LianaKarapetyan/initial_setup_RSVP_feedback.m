%% Settings for RSVP feedback
fb = struct();
fbint = struct();

fbint.fullscreen = 0; 
fbint.geometry = int16(VP_SCREEN);
fbint.symbol_vpos = 525;
fb.alphabet_vpos = 975;

fbint.word_vpos= 75;
fbint.word_font_size= 150;
fbint.word_target_font_size= 175;

fbint.font_size = 200;
%fbint.headline_font_size = 150;
%fbint.headline_target_font_size = 175;
fb.headline_margin_factor = 0.2;

%Display the current alphabet in corresponding colors 
fb.show_alphabet = 1;
fbint.alphabet_font_size= 100;

fb.inter_word = 0;
% Display time of the classifier-selected letter
fbint.present_eeg_input_time = 1;
fbint.countdown_start= 3;
%Allow the eeg input to be simulated by keyboard (for debug)
fb.allow_keyboard_input = 0;
%Display the countdown before each new word
fb.show_word_countdown = 0;
%Display the countdown before each new target
fb.show_trial_countdown = 0;

fbint.inter_trial_interval = 3;
fbint.sequences_per_trial = 10;

fbint.max_diff =20;
fbint.sound = 0;

fb.font_color_name = 'black';
fb.bg_color = 'grey';
fb.symbol_colors = {'red', 'white', 'blue','green','black'};
fb.color_groups = {'fRyGk&lt;', 'pJUX!E', 'iSwc-N','TBMqAH','LdvOz.'};
% mdash: 	U+2014 (8212)
fbint.nonalpha_trigger= {{'-',57}, {'.',58}, {'!',59}, {'&lt;',60}};
fb.meaningless = '';

fb.inter_burst =0.0;
fb.inter_sequence =0;
fb.inter_block =1;
fb.inter_trial =1;
fb.present_word_time= 2;
fb.present_target_time= 4.0;
fb.fixation_cross_time= 3.0;
fb.count_down_symbol_duration =.5;
fb.show_trial_fix_cross = 1;

fbint.trial_type= TRIAL_CALIBRATION;
fb.words= {phrase_practice};
fbint.alternating_colors= 1;
fb.symbol_duration= 0.103;  % effectively: 116ms

pyff('init','RSVPSpeller');
pause(2)

%% TESTING
%fbint.sequences_per_trial = 2;

pyff('set',fb);
pyff('setint',fbint);

pyff('save_settings', [acqFolder 'RSVP_Color116_feedback']);
% delete 'symbol_colors' from JSON file
