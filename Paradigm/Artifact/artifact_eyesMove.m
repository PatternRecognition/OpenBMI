function artifact_eyesMove(varargin)
% Example:
%    artifact_eyesMove({'soundDirectory',sound_dir; 'repeatTimes',10; 'blankTime',1.5; 'durationTime',1})

in=opt_cellToStruct(varargin{:});

%% Basic setting
white=[255 255 255];
in.repeatTimes=4*in.repeatTimes;
s_width=130;s_height=130;
nTr=1;

%% Load sound & image
[aud_up,freq] = audioread([in.soundDirectory '\speech_up.wav']);
[aud_down] = audioread([in.soundDirectory '\speech_down.wav']);
[aud_left] = audioread([in.soundDirectory '\speech_left.wav']);
[aud_right] = audioread([in.soundDirectory '\speech_right.wav']);
[aud_center] = audioread([in.soundDirectory '\speech_center.wav']);
nrchannels = 1;

%% Psychtoolbox
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('D010');
IO_LIB=which('inpoutx64.dll');


Screen('Preference', 'SkipSyncTests', 2);

screens=Screen('Screens');
screenNumber=max(screens);
gray=GrayIndex(screenNumber);
[w, wRect]=Screen('OpenWindow', 2, gray);

Screen('TextSize',w, 50);
DrawFormattedText(w, 'Click to start an experiment', 'center', 'center', [0 0 0]);
Screen('Flip', w);
GetClicks(w);

% Audio device
InitializePsychSound;
pahandle = PsychPortAudio('Open', [], [], 0, freq, nrchannels);


%% Start
% Order of stimuli (eyes movement)
% idx = randperm(in.repeatTimes);
eyesMove_task = repmat(1:4,[1,in.repeatTimes]);
% eyesMove_task = eyesMove_task(idx);
eyesMove_task = Shuffle(eyesMove_task);

% Start trigger
ppTrigger(111);

Screen('Flip', w);
WaitSecs(in.blankTime);

% Notification of start
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Move your eyes', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(3);

% Background image
draw_background
Screen('Flip', w);
WaitSecs(in.blankTime);

for i=1:in.repeatTimes
    switch eyesMove_task(i)
        case 1  % UP
            PsychPortAudio('FillBuffer', pahandle, aud_up');
            PsychPortAudio('Start',pahandle,1,0,1);
            
            draw_background
            Screen('FillRect', w, white, [wRect(3)/2-(s_width/2) wRect(1) wRect(3)/2+(s_width/2) wRect(1)+s_height]);
            Screen('Flip', w);
            ppTrigger(11);
            WaitSecs(in.durationTime);
            
        case 2  % DOWN
            PsychPortAudio('FillBuffer', pahandle, aud_down');
            PsychPortAudio('Start',pahandle,1,0,1);
            
            draw_background
            Screen('FillRect', w, white, [wRect(3)/2-(s_width/2) wRect(4)-s_height wRect(3)/2+(s_width/2) wRect(4)] );
            Screen('Flip', w);
            ppTrigger(13);
            WaitSecs(in.durationTime);
            
        case 3  % LEFT
            PsychPortAudio('FillBuffer', pahandle, aud_left');
            PsychPortAudio('Start',pahandle,1,0,1);
            
            draw_background
            Screen('FillRect', w, white, [wRect(1)  wRect(4)/2-(s_height/2)  wRect(1)+s_width   wRect(4)/2+(s_height/2)]);
            Screen('Flip', w);
            ppTrigger(15);
            WaitSecs(in.durationTime);
            
        case 4  % RIGHT
            PsychPortAudio('FillBuffer', pahandle, aud_right');
            PsychPortAudio('Start',pahandle,1,0,1);
            
            draw_background
            Screen('FillRect',w,white,[wRect(3)-s_width wRect(4)/2-(s_height/2) wRect(3) wRect(4)/2+(s_height/2)]);
            Screen('Flip', w);
            ppTrigger(17);
            WaitSecs(in.durationTime);
            
    end
    
    % CENTER
    PsychPortAudio('FillBuffer', pahandle, aud_center');
    PsychPortAudio('Start',pahandle,1,0,1);
    
    draw_background
    Screen('FillRect', w, white, FixCross');
    Screen('Flip', w);
    ppTrigger(19);
    WaitSecs(in.durationTime);
    
    disp(sprintf('Trial #%.d',nTr));
    nTr=nTr+1;
    
    
end

Screen('Flip', w);
WaitSecs(in.blankTime);

%% End
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(3);

% End trigger
ppTrigger(222);

Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);
