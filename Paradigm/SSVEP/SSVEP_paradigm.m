function SSVEP_paradigm_ambulatory(trig, nT, commun,scrNum,order_idx)
%% ear SSVEP experiment paradigm

%% parameter setting
% numTrial=50; % 50 * 3 stimulus = 150 trials (25  min)

% stimulus parameters
rest_nT = 15; % 몇번마다 쉴건지 
timeCue=1.5; timeSti=5; timeRest=2; 
% timeCue=0.5; timeSti=2; timeRest=1; % test
freq=[11 7 5];  % 5.45, 8.57, 12 Hz // 5-20 Hz 해야 됨  freq=[17 11 7 5]; 

% screen parameters
size_reduce = 0.5;
scrSize = [0, 0, 1920*size_reduce, 1080*size_reduce];%'full'; %screenNum=1;

% screen box
boxSize=150*size_reduce; btwBox=200*size_reduce; 
color=[255 255 255];

%% keys setting
startKey=KbName('space');
escapeKey = KbName('esc');
waitKey=KbName(']');

%% psychtoolbox
Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
black = BlackIndex(scrNum);
[w, wRect]=Screen('OpenWindow',scrNum, black,scrSize);
Screen('FillRect', w, black);
Textsize=50*size_reduce;%ceil(10);
Screen('TextSize',w,  ceil(10));
textbox = Screen('TextBounds', w, '+');

%% frequency
topPriorityLevel = MaxPriority(w);
ifi = Screen('GetFlipInterval', w);
num_frame = round(timeSti / ifi);
tot_frame = zeros(length(freq), num_frame);
for i = 1:length(freq)
    tot_frame(i, 1:freq(i):num_frame) = 1;
end
len_f = length(freq);

%% stimuli
% Order of stimuli (eyes movement)
% order_task = repmat(1:len_f,1,nT);
% order_task = repmat(1:3,1,20);
% order_task=Shuffle(order_task);

% 150 trial
% order_task=[3,1,2,3,1,3,1,1,3,1,3,2,1,3,2,1,3,2,1,3,3,1,2,2,1,3,2,2,3,2,1,2,1,2,3,1,3,2,3,1,1,3,1,1,2,1,2,3,1,2,3,3,3,3,2,1,3,3,2,1,3,1,3,2,1,3,1,1,2,2,1,1,2,1,2,1,2,2,1,1,3,1,2,2,3,1,2,1,3,3,1,2,2,1,3,3,2,2,3,1,2,2,1,1,2,3,3,3,3,1,2,2,3,2,3,1,1,2,1,1,3,3,1,3,2,3,2,2,2,2,1,2,3,3,2,3,2,3,3,1,3,1,3,2,2,3,2,1,2,1];
% 60 trial
order_task_list=[3,1,2,3,1,3,1,2,3,1,3,2,1,3,2,1,3,2,1,3,3,1,2,2,1,3,2,2,3,2,1,2,1,2,3,1,3,2,3,2,1,3,1,1,2,1,2,3,1,2,3,1,2,3,2,1,3,3,2,1;
    2,3,2,1,2,1,3,2,1,3,1,2,1,3,1,2,1,3,1,2,3,2,1,2,2,1,3,3,1,2,3,3,2,1,3,1,2,3,1,2,3,3,1,1,2,1,3,2,3,1,2,1,3,1,2,3,3,2,3,2;
    3,3,1,2,2,3,1,2,2,3,1,2,3,2,2,1,3,1,1,2,2,3,1,3,1,1,2,3,1,3,1,3,2,2,3,1,3,3,1,1,2,3,1,2,2,1,1,3,2,1,3,3,1,2,3,2,1,2,3,2;
    1,2,2,3,2,3,3,1,1,3,2,2,3,2,1,3,1,1,3,1,2,1,3,2,2,1,3,2,1,2,1,3,1,3,2,3,1,2,1,3,2,1,2,3,3,1,3,2,1,3,1,2,2,3,1,2,3,2,1,3];
order_task = order_task_list(order_idx,:);
% 40 trials = 4 classes * 10 trials
% stimuli=[1,2,4,3,1,3,4,2,1,3,4,2,1,2,4,3,1,2,3,4,1,3,1,2,4,3,1,2,3,1,4,1,4,2,4,1,3,2,4,3,1,2,3,1,4,1,2,4,2,3,4,3,4,2,3,1,4,2,1,2,3,4,2,1,3,1,2,4,3,1,2,3,4,2,4,3,1,4,3,2];

% nn = nT/20;
% order_task = repmat(stimuli,1,nn);
%% box(stimulus) position
%           1
%      2    3    4
%           5
wd=wRect(3); h=wRect(4);

if length(freq) == 4
% 1 2 4 5 위치
x1=[wd/2-boxSize/2, wd/2-3*boxSize/2-btwBox, wd/2+boxSize/2+btwBox, wd/2-boxSize/2];
x2=[wd/2+boxSize/2, wd/2-boxSize/2-btwBox, wd/2+3*boxSize/2+btwBox, wd/2+boxSize/2];
y1=[h/2-3*boxSize/2-btwBox, h/2-boxSize/2,h/2-boxSize/2, h/2+boxSize/2+btwBox];
y2=[h/2-boxSize/2-btwBox, h/2+boxSize/2, h/2+boxSize/2, h/2+3*boxSize/2+btwBox];
end

if length(freq) == 3
% 2 3 4 위치
x1=[wd/2-3*boxSize/2-btwBox, wd/2-boxSize/2, wd/2+boxSize/2+btwBox];
x2=[wd/2-boxSize/2-btwBox, wd/2+boxSize/2, wd/2+3*boxSize/2+btwBox];
y1=[h/2-boxSize/2, h/2-boxSize/2, h/2-boxSize/2];
y2=[h/2+boxSize/2, h/2+boxSize/2, h/2+boxSize/2];
end
%% beep
[speak_rest, fs_rest]=audioread('rest.mp3');
%% fixation cross
% [X,Y] = RectCenter(wRect);
% FixationSize = 40;
% FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% send trigger

tr_pause = 111;
tr_resume = 222;
tr_start = 11;
tr_end = 22;

% brain vision setting
if commun(1) == true
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('3010');
end


if commun(2) == true
% openviber setting
tcp_ear = tcpclient('localhost', 15361);
padding=uint64(0);
timestamp=uint64(0);

% ear trigger
stimulusPAUSE=[padding; uint64(tr_pause); timestamp];    % resume trigger
stimulusRESUME=[padding; uint64(tr_resume); timestamp];    % resume trigger
stimulusSTART=[padding; uint64(tr_start); timestamp];    % resume trigger
stimulusEND=[padding; uint64(tr_end); timestamp];    % resume trigger

stimulus1=[padding; uint64(trig(1)); timestamp];
stimulus2=[padding; uint64(trig(2)); timestamp];
stimulus3=[padding; uint64(trig(3)); timestamp];
% stimulus4=[padding; uint64(trig(4)); timestamp];

end

if commun(3) == true
% motion studio setting
s = serial('COM8');
set(s, 'BaudRate',9600,'DataBits',8,'parity','non','stopbits',1,'FlowControl','none');
fopen(s);
end


%% paradigm start
prevVbl = Screen('Flip',w);
disp('Press space to start SSVEP experiment')
[ keyIsDown, ~, keyCode ] = KbCheck;
while ~keyCode(startKey)
    Screen('TextSize',w, Textsize);
    DrawFormattedText(w, 'Press space to start SSVEP experiment ', 'center', 'center', [255 255 255]);
    [ keyIsDown, ~, keyCode ] = KbCheck;
    Screen('Flip', w);
end

Screen('TextSize',w, Textsize);
DrawFormattedText(w,'SSVEP task will start in 3 secs','center','center',[255 255 255]);
disp('SSVEP task will start in 3 secs')
Screen('Flip', w);
WaitSecs(1);
Screen('FillRect', w, black);
WaitSecs(2);
Screen('TextSize', w, ceil(10));

a1=0; a2=0; at=0;
% 75    115

if commun(2) == true 
    write(tcp_ear, stimulusSTART); end
if commun(1) == true
    ppWrite(IO_ADD,tr_start); end
if commun(3) == true
    fwrite(s,uint32(tr_start),'uint32'); end

for t=1:length(order_task)
    
    Priority(topPriorityLevel);
    %% Pause 실험 반 하고 좀 쉬어야지
%     if ( t == ceil(length(order_task)/2) ) %&& ( length(order_task) > 50 )
     if (rem(t,rest_nT)==1) 
        % rest
        if commun(2) == true
            write(tcp_ear, stimulusPAUSE); end
        if commun(1) == true
            ppWrite(IO_ADD,tr_pause); end
        if commun(3) == true
            %fwrite(s,uint32(tr_pause),'uint32');
        end
        
        if (t~=1)
            disp('Rest Time Press Space')
            sound(speak_rest,fs_rest);
            [ keyIsDown, ~, keyCode ] = KbCheck;
            while ~keyCode(startKey)
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w,'Break Time\n\nPress space to resume the experiment','center','center',[255 255 255]);
                Screen('Flip', w);
                [ keyIsDown, ~, keyCode ] = KbCheck;
            end
            
            Screen('TextSize',w, Textsize);
            disp('SSVEP task will resume in 3 secs')
            DrawFormattedText(w,'SSVEP task will resume in 3 secs','center','center',[255 255 255]);
            Screen('Flip', w);
            WaitSecs(3);
        end
        if commun(2) == true
            write(tcp_ear, stimulusRESUME); end
        if commun(1) == true
            ppWrite(IO_ADD,tr_resume); end
        if commun(3) == true
            %fwrite(s,uint32(tr_resume),'uint32');
        end
        
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
    Screen('FillRect', w, [255 255 0], [x1(ot),y1(ot),x2(ot),y2(ot)]);
    Screen('TextSize', w, ceil(10));
    for j=1:len_f
        Screen('DrawText', w, '+', x1(j)+boxSize/2 - textbox(3)/2,...
            y1(j)+boxSize/2 - textbox(4)/2, [128, 128, 128]);
    end
    Screen('Flip',w);     
    WaitSecs(timeCue);
    
    
    % flickering
    if commun(2) == true
        eval(sprintf('write(tcp_ear, stimulus%d)',ot)); end
    if commun(1) == true
        ppWrite(IO_ADD,trig(ot)); end
    if commun(3) == true
        fwrite(s,uint32(ot),'uint32'); end 
  
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
                if commun(2) == true
                    write(tcp_ear, stimulusEND); end
                if commun(1) == true
                    ppWrite(IO_ADD,tr_end); end
                if commun(3) == true
                    fwrite(s,uint32(tr_end),'uint32'); end
                
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(2);
                Screen('CloseAll');
                ShowCursor;
                fclose('all');
                Priority(0);
                return;
            elseif keyCode(waitKey)
                if commun(2) == true
                    write(tcp_ear, stimulusPAUSE); end
                if commun(1) == true
                    ppWrite(IO_ADD,tr_pause); end
                if commun(3) == true
                    %fwrite(s,uint32(tr_pause),'uint32');
                end
                while ~keyCode(startKey)
                    Screen('FillRect', w, black);
                    [ keyIsDown, ~, keyCode ] = KbCheck;
                    Screen('TextSize',w, Textsize);
                    DrawFormattedText(w, 'Pause experiment', 'center', 'center', [255 255 255]);
                    Screen('Flip', w);
                end
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w,'SSVEP task will resume in 3 secs','center','center',[255 255 255]);
                Screen('Flip', w);
                WaitSecs(3);
                
                if commun(2) == true
                    write(tcp_ear, stimulusRESUME); end
                if commun(1) == true
                    ppWrite(IO_ADD,tr_resume); end
                if commun(3) == true
                    %fwrite(s,uint32(tr_resume),'uint32');
                end
                
            end
        end
    end
    fprintf('%.4f \t %.4f\n', a2-a1, a1-at);
    
end

%% End
if commun(2) == true
    write(tcp_ear, stimulusEND); end
if commun(1) == true
    ppWrite(IO_ADD,tr_end); end
if commun(3) == true
    fwrite(s,uint32(tr_end),'uint32'); end

Screen('TextSize',w, Textsize);
DrawFormattedText(w,'The end of experiment','center','center',[255 255 255]);
Screen('Flip', w);
disp('END')

WaitSecs(2);
Screen('CloseAll');
ShowCursor;
fclose('all');
% delete(s);
% clear s
Priority(0);

