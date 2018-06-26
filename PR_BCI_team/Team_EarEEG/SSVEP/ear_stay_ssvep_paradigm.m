%% ear SSVEP experiment paradigm
clear all; close all; clc;

%% parameter setting
numTrial=20; % 20 * 3 stimulus = 60 trials -> 10s -> 600s

% stimulus parameters
timeSti=10; timeRest=2;
freq = [5,6,7];   % 12  8.57   6.67  5.45 Hz 
% ¿À¸í¼® [5,6,7];
% 5,6,7,8,9,11
% freq = [5, 7, 8, 9, 11];   % 12  8.57  7.5  6.67  5.45 Hz
boxSize=300; btwBox=150; 
btw_x=450;
btw_y=50;
color=[255 255 255];

% screen parameters
scrSize='full'; scrNum=2;


%% send trigger
% brain vision setting
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('E010');

% openviber setting
tcp_ear = tcpclient('localhost', 15361);
padding=uint64(0);
timestamp=uint64(0);
stimulusSTART=[padding; uint64(111); timestamp];    % start trigger
stimulusEND=[padding; uint64(222); timestamp];    % start trigger

stimulus1=[padding; uint64(1); timestamp];
stimulus2=[padding; uint64(2); timestamp];
stimulus3=[padding; uint64(3); timestamp];
stimulus4=[padding; uint64(4); timestamp];
stimulus5=[padding; uint64(5); timestamp];
stimulus11=[padding; uint64(11); timestamp];    % walking
stimulus21=[padding; uint64(21); timestamp];    % without walking
stimulus22=[padding; uint64(22); timestamp];    % without walking
stimulus23=[padding; uint64(23); timestamp];    % without walking

%% pause keys
startKey=KbName('space');
escapeKey = KbName('esc');
waitKey=KbName('*');
[speak_rest, fs_rest]=audioread('rest.mp3');
[speak_n{1}, fs_n{1}]=audioread('speech_1.wav');
[speak_n{2}, fs_n{2}]=audioread('speech_2.wav');
[speak_n{3}, fs_n{3}]=audioread('speech_3.wav');

%% psychtoolbox
screens = Screen('Screens');
black = BlackIndex(scrNum);
[w, wRect]=Screen('OpenWindow',scrNum, black);
Screen('FillRect', w, black);

%% frequency
topPriorityLevel = MaxPriority(w);
ifi = Screen('GetFlipInterval', w);
num_frame = round(timeSti / ifi);
tot_frame = zeros(4, num_frame);
for i = 1:length(freq)
    tot_frame(i, 1:freq(i):num_frame) = 1;
end
len_f = length(freq);

%% stimuli
% Order of stimuli (eyes movement)
order_task = repmat(1:len_f,1,numTrial);
order_task=Shuffle(order_task);

%% box(stimulus) position
%           1
%      2    3    4
%           5
wd=wRect(3); h=wRect(4);
if length(freq)==5
    x1=[wd/2-boxSize/2, wd/2-3*boxSize/2-btwBox, wd/2-boxSize/2, wd/2+boxSize/2+btwBox, wd/2-boxSize/2];
    x2=[wd/2+boxSize/2, wd/2-boxSize/2-btwBox, wd/2+boxSize/2, wd/2+3*boxSize/2+btwBox, wd/2+boxSize/2];
    y1=[h/2-3*boxSize/2-btwBox, h/2-boxSize/2, h/2-boxSize/2, h/2-boxSize/2, h/2+boxSize/2+btwBox];
    y2=[h/2-boxSize/2-btwBox, h/2+boxSize/2, h/2+boxSize/2, h/2+boxSize/2, h/2+3*boxSize/2+btwBox];
elseif length(freq)==4
    x1=[wd/2-boxSize/2, wd/2-3*boxSize/2-btwBox,  wd/2+boxSize/2+btwBox, wd/2-boxSize/2];
    x2=[wd/2+boxSize/2, wd/2-boxSize/2-btwBox, wd/2+3*boxSize/2+btwBox, wd/2+boxSize/2];
    y1=[h/2-3*boxSize/2-btwBox, h/2-boxSize/2,h/2-boxSize/2, h/2+boxSize/2+btwBox];
    y2=[h/2-boxSize/2-btwBox, h/2+boxSize/2, h/2+boxSize/2, h/2+3*boxSize/2+btwBox];
elseif length(freq)==3
    x1=[wd/2-3*boxSize/2-btw_x, wd/2-3*boxSize/2-btw_x, wd/2-3*boxSize/2-btw_x];
    x2=[wd/2-boxSize/2-btw_x, wd/2-boxSize/2-btw_x, wd/2-boxSize/2-btw_x];
    y1=[h/2+boxSize/2+btw_y, h/2-boxSize/2, h/2-3*boxSize/2-btw_y];
    y2=[h/2+3*boxSize/2+btw_y, h/2+boxSize/2, h/2-boxSize/2-btw_y];
end

%% beep
freq_beep=22000;
beepLengthSecs=0.5;
[beep,samplingRate] = MakeBeep(500,beepLengthSecs,freq_beep);
Snd('Open');

%% fixation cross
[X,Y] = RectCenter(wRect);
FixationSize = 40;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% start
prevVbl = Screen('Flip',w);

[ keyIsDown, ~, keyCode ] = KbCheck;
while ~keyCode(startKey)
    Screen('TextSize',w, 50);
    DrawFormattedText(w, 'Press space to start SSVEP experiment ', 'center', 'center', [255 255 255]);
    [ keyIsDown, ~, keyCode ] = KbCheck;
    Screen('Flip', w);
end

%% SSVEP without stop

[ keyIsDown, ~, keyCode ] = KbCheck;
while ~keyCode(startKey)
    Screen('TextSize',w, 50);
    DrawFormattedText(w, 'Press space to start SSVEP without walking', 'center', 'center', [255 255 255]);
    [ keyIsDown, ~, keyCode ] = KbCheck;
    Screen('Flip', w);
end

Screen('TextSize',w, 50);
DrawFormattedText(w,'SSVEP task without walking will start in 3 secs','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(1);
Screen('FillRect', w, black);
WaitSecs(2);
Screen('TextSize', w, ceil(10));
textbox = Screen('TextBounds', w, '+');

for t=1:length(order_task)
    disp(t);
    Priority(topPriorityLevel);
    
    %% show cue
    % background
    for j=1:len_f
        Screen('FillRect', w, color, [x1(j),y1(j),x2(j),y2(j)]);
    end
    Screen('TextSize', w, ceil(10));
    for j=1:len_f
        Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
            y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
    end
    
    % cue of this trial
    ot = order_task(t);
    sound(speak_n{ot}, fs_n{ot});
    Screen('FillRect', w, [255 255 0], [x1(ot),y1(ot),x2(ot),y2(ot)]);
    Screen('TextSize', w, ceil(10));
    for j=1:len_f
        Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
            y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
    end
    Screen('Flip',w);
    WaitSecs(2);
    
    
    % flickering
    Snd('Play',beep);
    eval(sprintf('write(tcp_ear, stimulus%d)',ot+20));
    ppWrite(IO_ADD,ot+20);
    
    a=GetSecs();
    for i = 1:num_frame % && run
        for n=1:size(tot_frame,1)
            if tot_frame(n, i)
                Screen('FillRect', w, color, [x1(n),y1(n),x2(n),y2(n)]);
            end
        end
        Screen('TextSize', w, ceil(10));
        for j=1:len_f
            Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
                y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
        end
        Screen('DrawingFinished', w);
        Screen('Flip',w);
    end
    disp(GetSecs() - a);
    Screen('TextSize', w, ceil(10));
    for j=1:len_f
        Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
            y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
    end
    Screen('Flip', w);
    
    sound(speak_rest,fs_rest);
    tic; %wait for escapekey or waitkey
    while toc < timeRest
        [ keyIsDown, ~, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(1);
                Screen('CloseAll');
                fclose('all');
                return;
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'stop experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                while ~keyCode(startKey)
                    Screen('FillRect', w, black);
                    [ keyIsDown, ~, keyCode ] = KbCheck;
                    Screen('TextSize',w, 50);
                    DrawFormattedText(w, 'Rest', 'center', 'center', [255 255 255]);
                    Screen('Flip', w);
                end
            end
        end
    end
end

DrawFormattedText(w,'The end of experiment','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(2);

Screen('CloseAll');
ShowCursor;
fclose('all');
output = 'Finish';
Priority(0);

