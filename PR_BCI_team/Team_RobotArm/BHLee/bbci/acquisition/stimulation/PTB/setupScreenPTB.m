% SETUPSCREENPTB -- initializes the screen. 
%
% Matthias Treder 24-08-2009


%% Setup OpenGL screen using PTB functions
if opt.fullscreen
    win=Screen('OpenWindow',opt.screenid); 
else
    win=Screen('OpenWindow',opt.screenid,[],VP_SCREEN); 
end
winRect = Screen('Rect', win);
slack = Screen('GetFlipInterval', win)/2;

