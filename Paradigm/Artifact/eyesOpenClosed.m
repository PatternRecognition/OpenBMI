function eyesOpenClosed(varargin)
% Example:
%    eyesOpenClosed({'soundDirectory',in.soundDirectory; 'repeatTimes',10; 'blankTime',1.5; 'durationTime',10})
in=opt_cellToStruct(varargin{:});
nTr=1;
%% Load sound
[aud_open,freq] = audioread([in.soundDirectory '\speech_eyes_open.wav']);
[aud_closed] = audioread([in.soundDirectory '\speech_eyes_closed.wav']);
[aud_stop] = audioread([in.soundDirectory '\speech_stop.wav']);

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

% Start trigger
ppTrigger(111);

for i=1:in.repeatTimes
    %% Eyes open
    Screen('Flip', w);
    WaitSecs(in.blankTime);
    
    % Notification of start
    Screen('TextSize',w, 50);
    DrawFormattedText(w, 'Open your eyes', 'center', 'center', [0 0 0]);
    Screen('Flip', w);
    
    WaitSecs(1);
    
    PsychPortAudio('FillBuffer', pahandle, aud_open');
    PsychPortAudio('Start', pahandle, 1,0,1);
    
    ppTrigger(61);
    WaitSecs(in.durationTime);
    
    PsychPortAudio('FillBuffer', pahandle, aud_stop');
    PsychPortAudio('Start', pahandle, 1,0,1);
%     ppTrigger(62);
    WaitSecs(0.5);
    
    disp(sprintf('Trial #%.d',nTr));
    nTr=nTr+1;

    %% Eyes closed
    Screen('Flip', w);
    WaitSecs(in.blankTime);
    
    % Notification of start
    Screen('TextSize',w, 50);
    DrawFormattedText(w, 'Close your eyes', 'center', 'center', [0 0 0]);
    Screen('Flip', w);
    
    WaitSecs(1);
    
    PsychPortAudio('FillBuffer', pahandle, aud_closed');
    PsychPortAudio('Start', pahandle, 1,0,1);
    
    ppTrigger(63);
    WaitSecs(in.durationTime);
    
    PsychPortAudio('FillBuffer', pahandle, aud_stop');
    PsychPortAudio('Start', pahandle, 1,0,1);
%     ppTrigger(64);
    WaitSecs(0.5);
    
    disp(sprintf('Trial #%.d',nTr));
    nTr=nTr+1;

end
Screen('Flip', w);
WaitSecs(3);

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
