function vis_oddball_paradigm_ambulatory(trig, nT, commun,screenNum,order_idx)



%% setting
% screen parameters
size_reduce = 0.5;
screenSize = [0, 0, 1920*size_reduce, 1080*size_reduce];%'full'; %screenNum=1;

timeRest = 5;
sti_duration = 0.5;
sti_size = 100*size_reduce;

%% psychtoolbox
% Screen('Preference', 'SkipSyncTests', 1);
% screens = Screen('Screens');
black = BlackIndex(screenNum);

if ischar(screenSize) && strcmp(screenSize,'full')
    [w, rect] = Screen('OpenWindow', screenNum );
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
else
    [w, rect] = Screen('OpenWindow', screenNum,[], screenSize);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
end

Screen('FillRect', w, black);
Textsize=50*size_reduce; %ceil(10);

Screen('TextSize',w,  ceil(10));
textbox = Screen('TextBounds', w, '+');

%% fixation cross
[X,Y] = RectCenter(rect);
FixationSize = 40*size_reduce;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% keys setting
startKey=KbName('space');
escapeKey = KbName('esc');
waitKey=KbName(']');

%%
[speak_rest, fs_rest]=audioread('rest.mp3');
%% beep
% freq_beep=22000;
% beepLengthSecs=0.5;
% [beep,samplingRate] = MakeBeep(2000,beepLengthSecs,freq_beep);
% Snd('Open');
%% stimuli make
sti_words = {'XXX','OOO'};
% t_interval = 1 + rand; % 1000 ms ~ 2000 ms random

% Order of stimuli (eyes movement)
if rem(nT,4) ~= 0
    error('NT should be multiple of 4');
    return;
end

% sti_unit = [1 1 1 1 2];
% nn = ceil(nT/length(sti_unit));
% stimuli = repmat(sti_unit,1,nn);
t_ratio = 0.2;
% stimuli = [repmat(2,1,nT*t_ratio) repmat(1,1,nT*(1-t_ratio))];
% order_task = Shuffle(stimuli);
% order_idx = Shuffle(1:5);
order_task_list = [1	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	1	1	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	1	1	2	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	1	1	2	1	2	1	1	1	1	1	2	1	1	1	1	2	1	1	1	2	1	2	1	1	1	2	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	1	2	1	1	1	1	2	1	1	1	2	1	2	1	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	2	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	1	1	1	2	1	1	1	2	1	1	1	1	1	1	1	2	1	2	1	1	1	2	1	1	2	1	1	1	1	1	1	2	1	1	1	1;
1	2	1	1	2	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	1	2	1	1	1	1	1	2	1	2	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	1	1	1	2	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	1	2	1	1	2	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	2	1	1	1	1	1	1	2	1	1	2	1	1	1	1	2	1	2	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	2	1	1	1	1	1	1	2	1	2	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	2	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	1	1	2	1	1	1	1	2	1	1	1	1	1	2	1	1;
1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	1	2	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	2	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	1	2	1	1	1	2	1	2	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	1	1	2	1	1	2	1	1	1	2	1	1	1	1	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	1	1	2	1	1	2	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	2	1	2	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	2	1	1	2	1	1	1	2	1	1	2	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	2	1	1	1	1	1	2	1	1	2	1	1;
1	2	1	1	2	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	1	2	1	2	1	1	1	2	1	1	2	1	1	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	2	1	2	1	1	1	1	1	1	1	2	1	1	1	2	1	1	2	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	2	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	2	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	2	1	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	1	1	2	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	2	1;
1	1	2	1	1	1	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	1	2	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	2	1	1	1	2	1	2	1	1	1	2	1	1	1	1	1	2	1	1	1	2	1	1	2	1	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	1	1	1	1	2	1	1	2	1	2	1	1	2	1	1	1	1	1	1	1	1	1	1	2	1	1	2	1	1	1	2	1	1	1	2	1	2	1	1	1	1	2	1	1	1	1	1	2	1	2	1	1	1	2	1	1	1	1	1	2	1	1	1	1	1	2	1	1	1	1	2	1	1	1	2	1	2	1	1	2	1	1	1	2	1	1	2	1	1	1	1	2	1	2	1	1	1	1	1	2	1	1	2	1	1	1	1	1	1	1	1	1	2	1	1	2	1	1];
order_task = order_task_list(order_idx,:);

%
% for rand_iter = 1:int8(8*rand()+1)
%   order_task = Shuffle(order_task);
% end


fprintf('\n    Total # of trial is %d\n',length(order_task))
% 100개 sample
%stimuli = [1,1,2,1,1,1,1,2,1,1,1,2,1,1,1,2,1,2,1,1,1,2,1,1,2,1,1,1,1,2,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,1,1,1,2,1,1,1,2,1,1,1,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,1,1,2,1,1];
% 600 trials
%nn = nT/100;
%order_task = repmat(stimuli,1,nn);

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

% openviber setting
if commun(2) == true
    tcp_ear = tcpclient('localhost', 15361);
    padding=uint64(0);
    timestamp=uint64(0);
    
    % ear trigger
    stimulusPAUSE=[padding; uint64(tr_pause); timestamp];    % resume trigger
    stimulusRESUME=[padding; uint64(tr_resume); timestamp];    % resume trigger
    stimulusSTART=[padding; uint64(tr_start); timestamp];    % resume trigger
    stimulusEND=[padding; uint64(tr_end); timestamp];    % resume trigger
    
    stimulus1=[padding; uint64(trig(1)); timestamp]; % non-target
    stimulus2=[padding; uint64(trig(2)); timestamp]; % target
end

% motion studio setting
if commun(3) == true
    s = serial('COM8');
    set(s, 'BaudRate',9600,'DataBits',8,'parity','non','stopbits',1,'FlowControl','none');
    fopen(s);
end

%% paradigm start
prevVbl = Screen('Flip',w);
disp('Press space to start Visual Oddball experiment ')
[ keyIsDown, ~, keyCode ] = KbCheck;
while ~keyCode(startKey)
    Screen('TextSize',w, Textsize);
    DrawFormattedText(w, 'Press space to start Visual Oddball experiment ', 'center', 'center', [255 255 255]);
    [ keyIsDown, ~, keyCode ] = KbCheck;
    Screen('Flip', w);
end
disp('Visual Oddball task will start in 3 secs')
Screen('TextSize',w, Textsize);
DrawFormattedText(w,'Visual Oddball task will start in 3 secs','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(2);
Screen('FillRect', w, black);
Screen('Flip', w);
WaitSecs(1);
Screen('TextSize', w, ceil(10));

%% stimuli start
nTrial = length(order_task);

if commun(2) == true 
    write(tcp_ear, stimulusSTART); end
if commun(1) == true
    ppWrite(IO_ADD,tr_start); end
if commun(3) == true
    fwrite(s,uint32(tr_start),'uint32'); end

Screen('TextSize',w, Textsize);
DrawFormattedText(w,'+','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(1);
nTarget = 0;
tic
for i=1:nTrial
    
    if (rem(i,100)==1) && (i~=1)
        % rest
        if commun(2) == true
            write(tcp_ear, stimulusPAUSE); end
        if commun(1) == true
            ppWrite(IO_ADD,tr_pause); end
        if commun(3) == true
            %fwrite(s,uint32(tr_pause),'uint32');
        end
        
        Screen('TextSize',w, Textsize);
        DrawFormattedText(w, 'Rest', 'center', 'center', [255 255 255]);
        Screen('Flip', w);
        sound(speak_rest,fs_rest);
        disp('Rest Time Press Space')
        [ keyIsDown, ~, keyCode ] = KbCheck;
        while ~keyCode(startKey)
            [ keyIsDown, ~, keyCode ] = KbCheck;
        end
        pause(1)

        if commun(2) == true
            write(tcp_ear, stimulusRESUME); end
        if commun(1) == true
            ppWrite(IO_ADD,tr_resume); end
        if commun(3) == true
            %fwrite(s,uint32(tr_resume),'uint32');
        end
        
    end
    
    if (rem(i,10)==1) && (i~=1)
        Screen('TextSize',w, Textsize);
        DrawFormattedText(w,'Rest\n\n Task will resume in 5 secs','center','center',[255 255 255]);
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
                    DrawFormattedText(w,'Visual Oddball task will resume in 3 secs','center','center',[255 255 255]);
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
        
%         Snd('Play',0.1*beep);
        Screen('TextSize',w, Textsize);
        DrawFormattedText(w,'+','center','center',[255 255 255]);   
        Screen('Flip', w);
        WaitSecs(2);
    end
    
    dur_time = toc;
    tic
    sti = order_task(i);
    show_word = sti_words{sti};
    
    Screen('TextSize',w, sti_size);
    DrawFormattedText(w,show_word,'center','center',[255 255 255]);
    Screen('Flip', w);
    
    
    % send trigger
    if commun(2) == true
        eval(sprintf('write(tcp_ear, stimulus%d)',sti)); end
    if commun(1) == true
        ppWrite(IO_ADD,trig(sti)); end
    if commun(3) == true
        fwrite(s,uint32(sti),'uint32'); end
    
    
    fprintf('Trial #%.d \t Target %.d \t 소요시간: %.4f \n',i, order_task(i), dur_time);
    nTarget = nTarget+ sti-1;
    fprintf('Number of Target %d\n', nTarget);
    
    WaitSecs(sti_duration);
    
    Screen('TextSize',w, Textsize);
    DrawFormattedText(w,'+','center','center',[255 255 255]);   
    Screen('Flip', w);
    
    t_interval = 0.5 + rand; % 500 ms ~ 1500 ms random
    WaitSecs(t_interval);
    
end

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
Priority(0);





