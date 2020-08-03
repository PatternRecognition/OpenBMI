%% Settings for RSVP feedback ('AlphaBurst')

fb = struct();
fbint = struct();

%% New variables for online speller

fb.alphabet_vpos = 1400;
fb.inter_word = .1;
% Display time of the classifier-selected letter
fbint.present_eeg_input_time = 1;
fbint.countdown_start = 3;
%Display the current alphabet in corresponding colors 
fb.show_alphabet = 0;
%Allow the eeg input to be simulated by keyboard (for debug)
fb.allow_keyboard_input = 1;
%Display the countdown before each new word
fb.show_word_countdown = 1;
%Display the countdown before each new target
fb.show_trial_countdown = 0;

%% Old variables offline speller
% static settings

fbint.inter_trial_interval = 3;
% fbint.sequences_per_trial = 10;
fbint.sequences_per_trial = 10;
fbint.fullscreen = 0; 
%  fbint.geometry = int16(VP_SCREEN);
fbint.geometry = int16([0  0  2560  1600]);
fbint.headline_vpos = 150;
fbint.symbol_vpos = 800;

fbint.font_size = 300;
fbint.headline_font_size = 200;
fbint.headline_target_font_size = 252;

% fbint.count_down_start =3;
fbint.max_diff =20;
fbint.sound = 0;

fb.headline_margin_factor = 0.2;
fb.font_color_name = 'black';

fb.symbol_colors = {'red', 'white', 'blue','green','black'};%test
% fb.symbol_colors = {'red', 'white', 'blue'};%test
fb.bg_color = 'grey';
% fb.color_groups = {'ABCDFGHIJ-', 'KMNWEQRST+', 'UVOXYZLP!/'};

%  ## Letter groups
% Uppercase 
% fb.color_groups = {'FDYGK&lt;', 'PJUX_E', 'ISWCZ/','TBMQAH','LRVON.'};%test
% % Lowercase
% fb.color_groups = {'fdygk-', 'pjux!e', 'iswcz/','tbmqah','lrvon.'};%test
% Mixed lower+uppercase
fb.color_groups = {'fRyGk&lt;', 'pJUX!E', 'iSwc_N','TBMqAH','LdvOz.'};%test
fbint.nonalpha_trigger= {{'_',57}, {'.',58}, {'!',59}, {'&lt;',60}};

% fb.meaningless = {'s','*+^%?;'};
fb.meaningless = '';

fb.inter_burst =0.0;
fb.inter_sequence =0.3;
fb.inter_block =1;
fb.inter_trial =1;
% fb.present_word_time =2;
fb.present_target_time = 4.0;
fb.fixation_cross_time =3.0;
fb.count_down_symbol_duration =.5;
fb.show_trial_fix_cross = 1;
        
%% photodiod test
% fb.symbol_colors = {'white','black'};%test
% fb.color_groups = {'O','Q'};%test
% fb.words={'OQ'};
% fbint.font_size = 750;

%%

% marker test settings
%fb_settings.symbol_colors = {'s', {'black', 'black', 'white'}};
%fb_settings.bg_color = {'s','black'};

% dynamic settings
fbint.alternating_colors= strcmp(color_mode, 'Color');
stim_durations= [.070 0.09 0.103 0.119];  % in s 
 idx= strmatch(speed_mode, {'83ms','100ms','116ms','133ms'},'exact');
if isempty(idx),
  error('unknown speed mode');
end
fb.symbol_duration= stim_durations(idx);
%fbint.alternating_colors=alternating_colors(cc(1));
%fb.symbol_duration= stim_durations(cc(2));
% fb.burst_duration = stim_durations(cc(2))*16;
%fb.words = words{current_condition(3)};

pyff('init','AlphaBurst');
pause(2)

%% TESTING
%fbint.sequences_per_trial = 2;

pyff('set',fb);
pyff('setint',fbint);
