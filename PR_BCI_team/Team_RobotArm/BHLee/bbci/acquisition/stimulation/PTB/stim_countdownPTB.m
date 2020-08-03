% STIM_COUNTDOWN - Presents a visual countdown on the screen. This function
% assumed that setupScreenPTB has been called before and that countdown is
% specified in opt.countdown.
%
% Note: Psychophysics toolbox (PTB-3) is needed to run this script! You can
% get the toolbox for free at http://psychtoolbox.org.
%
% Matthias Treder 25-08-2009

opt= set_defaults(opt, ...
                  'countdown', 3);

%% Text settings
%Screen('TextFont',win, 'Times');
Screen('TextSize',win, 80);
vbl = Screen('Flip', win);

for k=1:opt.countdown
    Screen('FillRect',win, opt.bg);       % Background
    DrawFormattedText(win,num2str(opt.countdown-k+1),'center','center',100);
%    Screen('DrawText', win, num2str(k),winRect(3)/2,winRect(4)/2);
    Screen('Flip',win);
    vbl = Screen('Flip',win,vbl + 1 - slack); % Wait 1 s
end
pause(1)
