%% Settings for pyff oddball (VisualOddball)
fb=struct();
fb.fullscreen= int16(1);
fb.screen_pos = int16(VP_SCREEN);

% Initialize oddball parameters
fb.FPS = 70;      % frames per sec
fb.nStim_per_block = 50;        
      

fb.show_standards = int16(1);
fb.give_feedback = int16(0);
fb.group_stim_markers = int16(0); 
            

% response options        
fb.rsp_key_dev = 'f'
fb.rsp_key_std = 'j' 
response_opts = ['none', 'dev_only', 'both'] % none: subject should not press a key
                                             % dev_only: subject should press only for deviants
                                             % both: subject response for both stds and devs        
fb.response = 'none'

% Durations of the different blocks
fb.feedback_duration= int16(300);
fb.stim_duration = 150;
fb.gameover_duration = int16(3000);
fb.shortpauseDuration = int16(10000);
fb.responsetime_duration = int16(100);
fb.beforestim_ival = [500, 600] - fb.stim_duration;  % randomly between the two values

% Feedback state booleans        
fb.gameover=int16(0);
fb.responsetime = int16(0);
fb.countdown=int16(1);
fb.firstStimTick = int16(1);
fb.beforestim=int16(0);
fb.shortpause = int16(0);
fb.feedback = int16(0);

% Initialize Visual Oddball parameters, 
fb.nStim = 180;   
fb.nStim_per_block = fb.nStim;   % no breaks
fb.dev_perc = 0.167; 
fb.dd_dist = 2 ;   % no contraint if deviant-deviant distance is 0 (cf. oddball sequence) 

shape_list={['square'],['circle'],['rectangle1'],['rectangle2'],['triangleup'],['triangledown'],['hourglass']}                        
fb.dev_shape=shape_list{conditions(ii,1)};
fb.dev_color=conditions(ii,2:4);
fb.std_color=[255,255,255];
fb.promptCount=int16(0);

pyff('init','VisualOddballNew');
pause(1);
pyff('set',fb)
pause(0.01);
