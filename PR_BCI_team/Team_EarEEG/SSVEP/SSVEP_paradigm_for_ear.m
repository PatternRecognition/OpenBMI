%% ear SSVEP experiment paradigm

%% parameter setting
numTrial=50; % 50 * 3 stim`ulus = 150 trials -> 8s/12s -> 1200s/1800s(30min)

% stimulus parameters
timeCue=1.5; timeSti=6; timeRest=2;
% timeCue=2; timeSti=8; timeRest=1.5;
% timeCue=1; timeSti=1; timeRest=2;
% freq = [5,6,7];   % 12  10  8.57 7.5 Hz 
freq=[6 7 8];  % 10, 8.57, 6.66667 Hz
% 오명석 [5,6,7];
% 5,6,7,8,9,11
% freq = [5, 7, 8, 9, 11];   % 12  8.57  7.5  6.67  5.45 Hz
boxSize=300; btwBox=150; 
btw_x=450;
btw_y=50;
color=[255 255 255];

% screen parameters
scrSize='full'; scrNum=1;


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
stimulusEND=[padding; uint64(222); timestamp];    % end trigger
stimulusPAUSE=[padding; uint64(11); timestamp];    % pause trigger
stimulusRESUME=[padding; uint64(22); timestamp];    % resume trigger

stimulus1=[padding; uint64(1); timestamp];
stimulus2=[padding; uint64(2); timestamp];
stimulus3=[padding; uint64(3); timestamp];


%% keys setting
startKey=KbName('space');
escapeKey = KbName('esc');
waitKey=KbName('*');

%% psychtoolbox
% Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
black = BlackIndex(scrNum);
[w, wRect]=Screen('OpenWindow',scrNum, black);
Screen('FillRect', w, black);
Textsize=50;%ceil(10);

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
% order_task = repmat(1:len_f,1,numTrial);
% order_task=Shuffle(order_task);
order_task=[3,1,2,3,1,3,1,1,3,1,3,2,1,3,2,1,3,2,1,3,3,1,2,2,1,3,2,2,3,2,1,2,1,2,3,1,3,2,3,1,1,3,1,1,2,1,2,3,1,2,3,3,3,3,2,1,3,3,2,1,3,1,3,2,1,3,1,1,2,2,1,1,2,1,2,1,2,2,1,1,3,1,2,2,3,1,2,1,3,3,1,2,2,1,3,3,2,2,3,1,2,2,1,1,2,3,3,3,3,1,2,2,3,2,3,1,1,2,1,1,3,3,1,3,2,3,2,2,2,2,1,2,3,3,2,3,2,3,3,1,3,1,3,2,2,3,2,1,2,1];

%% box(stimulus) position
%           1
%      2    3    4
%           5
wd=wRect(3); h=wRect(4);

% 2 3 4 위치
x1=[wd/2-3*boxSize/2-btwBox, wd/2-boxSize/2, wd/2+boxSize/2+btwBox];
x2=[wd/2-boxSize/2-btwBox, wd/2+boxSize/2, wd/2+3*boxSize/2+btwBox];
y1=[h/2-boxSize/2, h/2-boxSize/2, h/2-boxSize/2];
y2=[h/2+boxSize/2, h/2+boxSize/2, h/2+boxSize/2];


%% beep
freq_beep=22000;
beepLengthSecs=0.5;
[beep,samplingRate] = MakeBeep(500,beepLengthSecs,freq_beep);
Snd('Open');

%% fixation cross
[X,Y] = RectCenter(wRect);
FixationSize = 40;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% paradigm start
prevVbl = Screen('Flip',w);

[ keyIsDown, ~, keyCode ] = KbCheck;
while ~keyCode(startKey)
    Screen('TextSize',w, Textsize);
    DrawFormattedText(w, 'Press space to start SSVEP experiment ', 'center', 'center', [255 255 255]);
    [ keyIsDown, ~, keyCode ] = KbCheck;
    Screen('Flip', w);
end

Screen('TextSize',w, Textsize);
DrawFormattedText(w,'SSVEP task will start in 3 secs','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(1);
Screen('FillRect', w, black);
WaitSecs(2);
Screen('TextSize', w, ceil(10));
textbox = Screen('TextBounds', w, '+');

write(tcp_ear, stimulusSTART);
ppWrite(IO_ADD,111);
        write(tcp_ear, stimulusRESUME);
        ppWrite(IO_ADD,22);
a1=0; a2=0; at=0;
% 75    115

for t=1:length(order_task)
    
    Priority(topPriorityLevel);
    %% Pause 실험 반 하고 좀 쉬어야지
%     if ( t == ceil(length(order_task)/2) ) %&& ( length(order_task) > 50 )
     if (rem(t,15)==1) && (t~=1)
        write(tcp_ear, stimulusPAUSE);
        ppWrite(IO_ADD,11);
        [ keyIsDown, ~, keyCode ] = KbCheck;
        while ~keyCode(startKey)
            Screen('TextSize',w, Textsize);
            DrawFormattedText(w,'Break Time\n\nPress space to resume the experiment','center','center',[255 255 255]);
            Screen('Flip', w);
            [ keyIsDown, ~, keyCode ] = KbCheck;
        end
        write(tcp_ear, stimulusRESUME);
        ppWrite(IO_ADD,22);
        Screen('TextSize',w, Textsize);
        DrawFormattedText(w,'SSVEP task will resume in 3 secs','center','center',[255 255 255]);
        Screen('Flip', w);
        WaitSecs(3);
    end
        fprintf('Trial #%.d \t Target %.d \t',t, order_task(t));

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
    Snd('Play',beep);
    Screen('FillRect', w, [255 255 0], [x1(ot),y1(ot),x2(ot),y2(ot)]);
    Screen('TextSize', w, ceil(10));
    for j=1:len_f
        Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
            y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
    end
    Screen('Flip',w);
    WaitSecs(timeCue);
    
    
    % flickering
    eval(sprintf('write(tcp_ear, stimulus%d)',ot));
    ppWrite(IO_ADD,ot);
    
    at=a1;
    a1=GetSecs();
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
    a2=GetSecs();
    Screen('TextSize', w, ceil(10));
    for j=1:len_f
        Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
            y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
    end
    Screen('Flip', w);
    
    tic; %wait for escapekey or waitkey
    while toc < timeRest
        [ keyIsDown, ~, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(2);
                write(tcp_ear, stimulusEND);
                ppWrite(IO_ADD,222);
                Screen('CloseAll');
                ShowCursor;
                fclose('all');
                Priority(0);
                return;
            elseif keyCode(waitKey)
                write(tcp_ear, stimulusPAUSE);
                ppWrite(IO_ADD,11);
                while ~keyCode(startKey)
                    Screen('FillRect', w, black);
                    [ keyIsDown, ~, keyCode ] = KbCheck;
                    Screen('TextSize',w, Textsize);
                    DrawFormattedText(w, 'Pause experiment', 'center', 'center', [255 255 255]);
                    Screen('Flip', w);
                end
                write(tcp_ear, stimulusRESUME);
                ppWrite(IO_ADD,22);
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w,'SSVEP task will resume in 3 secs','center','center',[255 255 255]);
                Screen('Flip', w);
                WaitSecs(3);
            end
        end
    end
          fprintf('%.4f \t %.4f\n', a2-a1, a1-at);

end

%% End
Screen('TextSize',w, Textsize);
DrawFormattedText(w,'The end of experiment','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(2);
write(tcp_ear, stimulusEND);
ppWrite(IO_ADD,222);
Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);

