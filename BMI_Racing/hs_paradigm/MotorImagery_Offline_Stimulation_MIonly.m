clear all; close all; clc;
%----------------------------------------------------------------------
%                       EEG Device setup
%----------------------------------------------------------------------
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('C010');
% IO_ADDR=hex2dec('E010');
IO_LIB=which('inpoutx64.dll');

runs = [60 45 45];
%----------------------------------------------------------------------
%                       Screen setup
%----------------------------------------------------------------------
% Here we call some default settings for setting up Psychtoolbox
% PsychDefaultSetup(2);

% Skip sync tests for demo purposes only
Screen('Preference', 'SkipSyncTests', 1);

% Get the screen numbers
screens = Screen('Screens');
% Draw to the external screen if avaliable
screenNumber = max(screens);
% Define black and white
gray=GrayIndex(screenNumber);
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);

% Open an on screen window
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, gray);

% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% % Query the frame duration
ifi = Screen('GetFlipInterval', window);

% % Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

%image load
left_arrow=imread('stimuli\MotorImagery Stimulation Images\left_arrow.jpg');
right_arrow=imread('stimuli\MotorImagery Stimulation Images\right_arrow.jpg');
up_arrow=imread('stimuli\MotorImagery Stimulation Images\up_arrow.jpg');
rest_arrow=imread('stimuli\MotorImagery Stimulation Images\rest_square.jpg');
pause_image=imread('stimuli\MotorImagery Stimulation Images\rest.jpg');

%----------------------------------------------------------------------
%                    Timing and Order Information
%----------------------------------------------------------------------
% interval setting
RepeatTimes = 60;

ref_times = 2;
cue_times = 4;
pause_times = 2;

% provide the random tasks
while(1)
    sti_task=randi([1 3], RepeatTimes, 1);
    if length(find(sti_task(:,1)==1))==length(find(sti_task(:,1)==2))
        if length(find(sti_task(:,1)==1))==length(find(sti_task(:,1)==3))
            break;
        end
    end
end

%----------------------------------------------------------------------
%                        Fixation Cross
%----------------------------------------------------------------------
% Here we set the size of the arms of our fixation cross
fixCrossDimPix = 50;

% Now we set the coordinates (these are all relative to zero we will let
% the drawing routine center the cross in the center of our monitor for us)
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];

% Set the line width for our fixation cross
lineWidthPix = 4;

%----------------------------------------------------------------------
%                      Experimental Loop
%----------------------------------------------------------------------
% Wait for mouse click:
Screen('TextFont', window, 'Arial');
Screen('TextSize',window, 24);   % Setup the text type for the window
DrawFormattedText(window, 'Mouse click to start MotorImagery experiment', ...
    'center', 'center', 0); % '1': white color
Screen('Flip', window);
GetClicks(window);
ppTrigger(5) %START

for r = 1:length(runs)
    RepeatTimes = runs(r);

    ref_times = 2;
    cue_times = 4;
    pause_times = 2;

    % provide the random tasks
    while(1)
        sti_task=randi([1 3], RepeatTimes, 1);
        if length(find(sti_task(:,1)==1))==length(find(sti_task(:,1)==2))
            if length(find(sti_task(:,1)==1))==length(find(sti_task(:,1)==3))
                break;
            end
        end
    end

    for i=1:RepeatTimes
        i
        % Draw the fixation cross in white, set it to the center of our screen and
        % set good quality antialiasing
        Screen('DrawLines', window, allCoords,...
             lineWidthPix, black, [xCenter yCenter], 2);
        % Convert our current number to display into a string
    %     numberString = num2str(i);
    % 
    %     % Draw our number to the screen
    %     DrawFormattedText(window, numberString, 'center', 'center', 1);
        % Flip to the screen. This command basically draws all of our previous
        % commands onto the screen. See later demos in the animation section on more
        % timing details. And how to demos in this section on how to draw multiple
        % rects at once.
        [vbl startrt]=Screen('Flip', window);

    %     WaitSecs(ref_times);    % Wait until [ref_times (2 s)] secs after last stimulus onset

        WaitSecs(1);
        beep
        WaitSecs(1);

       %% resting (4s)
        tex=Screen('MakeTexture', window, rest_arrow); 
        Screen('DrawTexture', window, tex);
        Screen('DrawLines', window, allCoords,...
            lineWidthPix, black, [xCenter yCenter], 2);
        Screen('Close',tex);
        vbl = Screen('Flip', window);
        ppTrigger(4);   % rest

        cue = sti_task(i,1);
        WaitSecs(cue_times);
        ppTrigger(7);
        
       %% Motor Imagery (4s)
        switch sti_task(i,1)
            case 1            
                tex=Screen('MakeTexture', window, left_arrow);
                Screen('DrawTexture', window, tex);
                Screen('DrawLines', window, allCoords,...
                    lineWidthPix, black, [xCenter yCenter], 2);
                Screen('Close',tex);
                vbl = Screen('Flip', window);
                ppTrigger(1);   % left
            case 2
                tex=Screen('MakeTexture', window, right_arrow);
                Screen('DrawTexture', window, tex);
                Screen('DrawLines', window, allCoords,...
                    lineWidthPix, black, [xCenter yCenter], 2);
                Screen('Close',tex);
                vbl = Screen('Flip', window);
                ppTrigger(2);   % right
            case 3
                tex=Screen('MakeTexture', window, up_arrow);
                Screen('DrawTexture', window, tex);
                Screen('DrawLines', window, allCoords,...
                    lineWidthPix, black, [xCenter yCenter], 2);
                Screen('Close',tex);
                vbl = Screen('Flip', window);
                ppTrigger(3);   % foot or both hand
        end

        cue = sti_task(i,1);
        WaitSecs(cue_times);
        ppTrigger(7);

    end
    
    beep
    WaitSecs(120);  % 2 min resting
    
end

WaitSecs(2);

% finish
ppTrigger(6);   % END
Screen('TextSize',window, 24);
DrawFormattedText(window, 'Thank you.', 'center', 'center', 1);
WaitSecs(3);
Screen('Flip', window); % Flip to the screen

Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);