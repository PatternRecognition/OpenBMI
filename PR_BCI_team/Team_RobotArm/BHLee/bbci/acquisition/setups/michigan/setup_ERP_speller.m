% pyff('init','HexoSpellerVE');
pyff('init','CakeSpellerVE');
pause(1);
pyff('load_settings', SPELLER_file);
pause(1);
pyff('setint', 'screenPos', VP_SCREEN);

return

%%
% -- OBSOLETE
%

% Feedback settings for the spellers 
fb= struct();
fbint= struct();

fbint.offline= (offline_mode);
fbint.debug= (0);

% Visual settings
fbint.screenWidth=  1900;
fbint.screenHeight= 1200;        
fbint.screenPos= VP_SCREEN;

fbint.wordboxWidth = 800;
fbint.fullscreen = 1;
fb.stimuli_colors = {[1.0 0.0 0.0], ...
                     [0.0 0.8 0.0], ...
                     [0.0 0.0 1.0], ...
                     [1.0 1.0 0.0], ...
                     [1.0 0.0 0.7], ...
                     [0.9 0.9 0.9]};
fb.shape_color = [0 0 0];
fb.letter_color = [.8 .8 .8];
fbint.use_oscillator = 1;

% Eyetracker
fbint.use_eyetracker = 0;
fbint.et_duration = 100;
fbint.et_range = 200;       % Maximum acceptable distance between target and actual fixation
fbint.et_range_time = 200; 

% Timing and trials
tframe = 0.016;    % time for one frame
fb.stimulus_duration = .1-tframe *.5;  % subtract 0.5*tframe to prevent frameskips
fb.interstimulus_duration =.1-tframe *.5;  % ISI
fb.animation_time = .7-tframe *.5;  % 1.5-tframe *.1;
fb.feedback_duration = .3;   % 1.
fb.wait_before_classify = 1;

fb.nCountdown = (2);
fbint.nr_sequences = (10);  % 10
fbint.min_dist = (2);

%% Init Speller and send settings
pyff('init','HexoSpellerVE');pause(1.5)
pyff('set',fb);
pyff('setint',fbint);

pyff('save_settings', [BCI_DIR 'acquisition\setups\michigan\HexoSpellerVE']);

% 
% 
%         self.shapes = [['triangle',  {'size':200}],
%                        ['rectangle', {'size':(180., 50), 'orientation':45.}],
%                        ['rectangle', {'size':(180., 40), 'orientation':-45.}],
%                        ['triangle',  {'size':200, 'orientation':180.}],
%                        ['hourglass', {'size':100}],
%                        ['circle',    {'radius':90}]]