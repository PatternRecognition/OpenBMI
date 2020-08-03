switch(cs)
  case 'overt_cake'  
    spellerName = 'CakeSpellerMVEP';
  case 'covert_cake'
    spellerName = 'CakeSpellerMVEP';
  case 'CenterSpellerMVEP'
    spellerName = 'CenterCakeSpellerMVEP';
    fb.letter_color = ones(1,3)*0.5;
end

pyff('init',spellerName);

%% Feedback settings for the three spellers 
fbint = struct();   % All variables that are integer type
fb= struct();       % All other variables

%% For DDebugging:
fb.debug= int16(0);

fbint.offline= offline_mode;

% Visual settings
fbint.screenPos = VP_SCREEN;
fbint.fullscreen = 1;
fb.letter_color = [1 1 1];
fb.phrase_color= 0.6*[1 1 1];
fb.current_letter_color= [1 1 1];

% Eyetracker (we need eyetracker only for the covert cake condition!)
fbint.use_eyetracker = strcmp(cs,'covert_cake'); 
%fbint.use_eyetracker = 1; 
fbint.et_duration = 100;
fbint.et_range = 200;       % Maximum acceptable distance between target and actual fixation
fbint.et_range_time = 200; 

% Timing and trials
tframe = 0.016;    % time for one frame
fb.stimulus_duration = .1 - tframe *.9;  % subtract 0.8*frame to prevent frameskips
fb.interstimulus_duration = .1 - tframe *.9;  % ISI
fb.intertrial_duration = 0; 
fb.animation_time = 1.2 - tframe *.9;
fb.nCountdown = 3;
fbint.nr_sequences = 10;
fbint.min_dist = 2;

% Motion VEP parameters
fb.countdown_color = [0.8, 0.8, 0.8];

if strcmp(spellerName,'CakeSpellerMVEP')
  % motion onset stimulus parameters
  % start/end relative to speller radius
  fb.onset_bar_start = 0.47;%0.5;
  fb.onset_bar_end = 0.53;%0.8;
  fb.onset_bar_width = 5.0;
  fb.onset_color = [0.6, 0.6, 0.6];
  % how fine the animation should be resolved temporally
  fbint.soa_steps = ceil(fb.stimulus_duration/tframe);
  % enlarge motion-onset bar while moving
  fbint.enlarge_bar = 0;
  % if set to true letters are postioned along the hexagon
  % outlines
  fbint.alt_letter_layout = 1;
  fbint.alt_speller_radius = 320;
  fbint.alt_font_size_level2 = 70;

  % if set to true each triangle will have one fixation point
  fb.alt_fix_points = ~strcmp(cs,'covert_cake');
  % distance from middle (proportional to radius)
  fb.alt_fix_points_dist = 0.5;
  % disable fix points at inter-trial period
  fbint.alt_fix_points_disable_inter = 0;
else
  fb.stripes_speed = 0.05;
	fb.stripes_color = .7 * [1 1 1];
  fb.stripes_angle = .8;       % angle of arrows (0.5 -> 90 degrees)
  fbint.stripes_count = 12;       % number of stripes
  fb.stripes_distance = 30.0;
  if strcmp(spellerName,'CenterCakeSpellerMVEP')
    fb.stripes_line_width = 2.0;
  end
end
%%
pyff('set',fb);
pyff('setint',fbint);