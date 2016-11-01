function artifact_theRest(varargin)
% Example:
%    artifact_theRest({'blankTime',3;'durationTime',5})

in=opt_cellToStruct(varargin{:});

%% Basic setting
freq=22000;
beepLengthSecs=0.5;
Beep = MakeBeep(500, beepLengthSecs, freq);
nrchannels = 1;

%% Load image
img_start=imread('\Stimulus\start.jpg');
img_blink=imread('\Stimulus\blink.jpg');
img_teeth=imread('\Stimulus\\teeth.jpg');
img_shoulder=imread('\Stimulus\shoulder.jpg');
img_head=imread('\Stimulus\head.jpg');

%% Load sound
[aud_stop] = audioread([in.soundDirectory '\speech_stop.wav']);

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
%% Start
image=img_start;
tex1=Screen('MakeTexture', w, image );
Screen('DrawTexture', w, tex1);
WaitSecs(7)

%% Blink eyes
Screen('Flip', w);
WaitSecs(in.blankTime);

% Notification of start
image=img_blink;
tex1=Screen('MakeTexture', w, image );
Screen('DrawTexture', w, tex1);
% Screen('TextSize',w, 50);
% DrawFormattedText(w, 'Blink your eyes\n\nafter beep sound', 'center', 'center', [0 0 0]);
Screen('Flip', w);

WaitSecs(1);

PsychPortAudio('FillBuffer', pahandle, Beep);
PsychPortAudio('Start', pahandle, 1,0,1);

ppTrigger(21);
WaitSecs(in.durationTime);
    PsychPortAudio('FillBuffer', pahandle, aud_stop');
    PsychPortAudio('Start', pahandle, 1,0,1);
    WaitSecs(0.3);
% ppTrigger(22);

%% Clench teeth
Screen('Flip', w);
WaitSecs(in.blankTime);

% Notification of start
image=img_teeth;
tex1=Screen('MakeTexture', w, image );
Screen('DrawTexture', w, tex1);
% Screen('TextSize',w, 50);
% DrawFormattedText(w, 'Clench your teeth\n\nafter beep sound', 'center', 'center', [0 0 0]);
Screen('Flip', w);

WaitSecs(1);

PsychPortAudio('FillBuffer', pahandle, Beep);
PsychPortAudio('Start', pahandle, 1,0,1);

ppTrigger(31);
WaitSecs(in.durationTime);
    PsychPortAudio('FillBuffer', pahandle, aud_stop');
    PsychPortAudio('Start', pahandle, 1,0,1);
    WaitSecs(0.3);
% ppTrigger(32);

%% Lift shoulders
Screen('Flip', w);
WaitSecs(in.blankTime);

% Notification of start
image=img_shoulder;
tex1=Screen('MakeTexture', w, image );
Screen('DrawTexture', w, tex1);
% Screen('TextSize',w, 50);
% DrawFormattedText(w, 'Lift your shoulders\n\nafter beep sound', 'center', 'center', [0 0 0]);
Screen('Flip', w);

WaitSecs(1);

PsychPortAudio('FillBuffer', pahandle, Beep);
PsychPortAudio('Start', pahandle, 1,0,1);

ppTrigger(41);
WaitSecs(in.durationTime);
    PsychPortAudio('FillBuffer', pahandle, aud_stop');
    PsychPortAudio('Start', pahandle, 1,0,1);
    WaitSecs(0.3);
% ppTrigger(42);

%% Head movement
Screen('Flip', w);
WaitSecs(in.blankTime);

% Notification of start
image=img_head;
tex1=Screen('MakeTexture', w, image );
Screen('DrawTexture', w, tex1);
% Screen('TextSize',w, 50);
% DrawFormattedText(w, 'Move your head from side to side\n\nafter beep sound', 'center', 'center', [0 0 0]);
Screen('Flip', w);

WaitSecs(1);

PsychPortAudio('FillBuffer', pahandle, Beep);
PsychPortAudio('Start', pahandle, 1,0,1);

ppTrigger(51);
WaitSecs(in.durationTime);
    PsychPortAudio('FillBuffer', pahandle, aud_stop');
    PsychPortAudio('Start', pahandle, 1,0,1);
    WaitSecs(0.3);
% ppTrigger(52);

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
