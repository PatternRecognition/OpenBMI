function output = Makeparadigm_SSVEP_manual (varargin)
fclose('all');
% Makeparadigm_SSVEP (Experimental paradigm):
%
% Description:
%   Basic SSVEP experiment paradigm using psychtoolbox.
%   It shows flickering boxes. Before flickering, random target is indicated in yellow.
%
% Example:
%   Makeparadigm_SSVEP ({'time_sti',5;'num_trial',10;'time_rest',3;'freq',[7.5 10 12 15 20];'boxSize',150;'betweenBox',200});
%
% Input: (Nx2 size, cell-type)
%   time_sti   - time for a stimulus [s]
%   time_rest  - time for rest, showing nothing but black screen [s]
%   color      - RGB color of stimulus box (e.g.[255 255 255])
%   num_trial  - number of trials per class
%   freq       - number of classes you want, from 1 to 5 (up,left,center,right,down)
%   boxSize    - size of stimulus box [pixel]
%   betweenBox - distance between stimulus boxes [pixel]
%   num_screen  - number of screen to show
%   screen_size - size of window showing experimental stimulus
%                 'full' or matrix (e.g.[0 0 300 300])
% Trigger Information
%   111             - starting the paradigm
%   1 - # of trials - stimulus
%   19              - checking the connection with analysis
%   222             - finish
% Made by hkkim

%% variables
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'port'), error('No input port information');
else port= opt.port;
end
f=opt.freq;
if isfield(opt,'color'),color = opt.color;
else color=[255 255 255];
end
n1=0;n2=0;n3=0;n4=0;n5=0;
escapeKey = KbName('esc');
waitKey=KbName('*');
res = 0;
%% trigger setting
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(port);

%% TCPIP
% global sock;
if opt.online
    % server eeg
    sock = tcpip('localhost', 3000, 'NetworkRole', 'Server');
    set(sock, 'InputBufferSize', 1024);
    % Open connection to the client
    fprintf('%s \n','Client Connecting...');
    fopen(sock)
    fprintf('%s \n','Client Connected');
    connectionServer = sock;
    set(connectionServer,'Timeout',2);
end
% port=12300;
% sock = tcpip('0.0.0.0', port, 'NetworkRole', 'server', 'timeout', .2);
% fopen(sock);
% for i = 1:5
%     [trigger, a] = fread(sock,1);
%     if(a > 0)
%         break;
%     end
% end
% ppTrigger(trigger);
% set(handles.noti_text, 'String','Connection');
%% Check the tcp/ip connection
% flushinput(sock);
% for i = 1:5
%     tmp = fread(sock,1);
%     if isempty(fread(sock,1)) && i == 5
%         output = 'Check your connection';
%         return;
%     elseif tmp == 19
%         break;
%     end
% end
% flushinput(sock);
% fwrite(sock,length(f)*opt.num_trials);
% output = 'EEERRRRR';

%% psychtoolbox
screens = Screen('Screens');
if ~isfield(opt,'num_screen')||(opt.num_screen > max(screens)),num_screen=max(screens); else num_screen=opt.num_screen; end
if ~isfield(opt,'screen_size'),screen_size='full'; else screen_size=opt.screen_size; end
black = BlackIndex(num_screen);
if isequal(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',num_screen, black);
else
    [w, wRect]=Screen('OpenWindow',num_screen, black, screen_size);
end
Screen('FillRect', w, black);

%% frequency
topPriorityLevel = MaxPriority(w);
ifi = Screen('GetFlipInterval', w);
num_frame = round(opt.time_sti / ifi);
tot_frame = zeros(4, num_frame);
opt.freq = [5, 7, 9, 11];
for i = 1:length(f)
    tot_frame(i, 1:f(i):num_frame) = 1;
end
len_f = length(f);

%% stimuli
% Order of stimuli (eyes movement)
order_task = repmat(1:len_f,1,opt.num_trials);
order_task=Shuffle(order_task);

%% box(stimulus) position
%           1
%      2    3    4
%           5
s_size=opt.boxSize; wd=wRect(3); h=wRect(4); inter=opt.betweenBox;
x1=[wd/2-s_size/2, wd/2-3*s_size/2-inter, wd/2+s_size/2+inter, wd/2-s_size/2];
x2=[wd/2+s_size/2, wd/2-s_size/2-inter, wd/2+3*s_size/2+inter, wd/2+s_size/2];
y1=[h/2-3*s_size/2-inter, h/2-s_size/2, h/2-s_size/2, h/2+s_size/2+inter];
y2=[h/2-s_size/2-inter, h/2+s_size/2, h/2+s_size/2, h/2+3*s_size/2+inter];

%% start
prevVbl = Screen('Flip',w);
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Mouse click to start SSVEP experiment ', 'center', 'center', [255 255 255]);
Screen('Flip', w);
GetClicks(w);

%% beep
freq=22000;
beepLengthSecs=0.5;
[beep,samplingRate] = MakeBeep(500,beepLengthSecs,freq);
Snd('Open');

%% fixation cross
[X,Y] = RectCenter(wRect);
FixationSize = 40;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% Resting State
DrawFormattedText(w,'Closed your eyes\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,77);
Screen('Flip', w);
WaitSecs(opt.rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'Recording Resting state\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,78);
Screen('FillRect', w, [255 255 255], FixCross');
Screen('Flip', w);
WaitSecs(opt.rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'It will start in 3 secs','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
WaitSecs(3);
Screen('Flip', w);

% ppTrigger(111);
ppWrite(IO_ADD,111);
WaitSecs(4);
Screen('TextSize', w, ceil(10));
textbox = Screen('TextBounds', w, '+');
for t=1:length(order_task)
    disp(t);
    if length(order_task)/2 == t
        Screen('TextSize',w, 50);
        DrawFormattedText(w,'Rest\n\n(Pause the brain vision)','center','center',[255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
        DrawFormattedText(w,'(Resume recording)','center','center',[255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
        DrawFormattedText(w,'Click to continue the experiment','center','center',[255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
    end
    Priority(topPriorityLevel);
    
    %% show cue
    % background
    for j=1:len_f
        Screen('FillRect', w, color, [x1(j),y1(j),x2(j),y2(j)]);
    end
    Screen('TextSize', w, ceil(10));
    Screen('DrawText', w, '+', x1(1)+s_size/2 - textbox(3)/2,...
        y1(1)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(2)+s_size/2 - textbox(3)/2,...
        y1(2)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(3)+s_size/2 - textbox(3)/2,...
        y1(3)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(4)+s_size/2 - textbox(3)/2,...
        y1(4)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    % cue of this trial
    Snd('Play',beep);
    ot = order_task(t);
    Screen('FillRect', w, [255 255 0], [x1(ot),y1(ot),x2(ot),y2(ot)]);
    Screen('TextSize', w, ceil(10));
    Screen('DrawText', w, '+', x1(1)+s_size/2 - textbox(3)/2,...
        y1(1)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(2)+s_size/2 - textbox(3)/2,...
        y1(2)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(3)+s_size/2 - textbox(3)/2,...
        y1(3)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(4)+s_size/2 - textbox(3)/2,...
        y1(4)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('Flip',w);
    ppWrite(IO_ADD,15);
    WaitSecs(4);
    ppWrite(IO_ADD,ot);
    %     ppTrigger(ot);
    %     switch order_task(t)
    %         case 1
    %             Screen('FillRect', w, [255 255 0], [x1(1),y1(1),x2(1),y2(1)]);
    %             Screen('Flip',w);
    %             WaitSecs(2);
    %             ppTrigger(1)
    %             n1=n1+1;
    %         case 2
    %             Screen('FillRect', w, [255 255 0], [x1(2),y1(2),x2(2),y2(2)]);
    %             Screen('Flip',w);
    %             WaitSecs(2);
    %             ppTrigger(2)
    %             n2=n2+1;
    %         case 3
    %             Screen('FillRect', w, [255 255 0], [x1(3),y1(3),x2(3),y2(3)]);
    %             Screen('Flip',w);
    %             WaitSecs(2);
    %             ppTrigger(3)
    %             n3=n3+1;
    %         case 4
    %             Screen('FillRect', w, [255 255 0], [x1(4),y1(4),x2(4),y2(4)]);
    %             Screen('Flip',w);
    %             WaitSecs(2);
    %             ppTrigger(4)
    %             n4=n4+1;
    %         case 5
    %             Screen('FillRect', w, [255 255 0], [x1(5),y1(5),x2(5),y2(5)]);
    %             Screen('Flip',w);
    %             WaitSecs(2);
    %             ppTrigger(5)
    %             n5=n5+1;
    %     end
    %
    % stimuli
    a=GetSecs();
    for i = 1:num_frame % && run
        for n=1:size(tot_frame,1)
            if tot_frame(n, i)
                Screen('FillRect', w, color, [x1(n),y1(n),x2(n),y2(n)]);
            end
        end
        Screen('TextSize', w, ceil(10));
        Screen('DrawText', w, '+', x1(1)+s_size/2 - textbox(3)/2,...
            y1(1)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawText', w, '+', x1(2)+s_size/2 - textbox(3)/2,...
            y1(2)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawText', w, '+', x1(3)+s_size/2 - textbox(3)/2,...
            y1(3)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawText', w, '+', x1(4)+s_size/2 - textbox(3)/2,...
            y1(4)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawingFinished', w);
        Screen('Flip',w);
    end
    disp(GetSecs() - a);
    ppWrite(IO_ADD,14);
    %     ppTrigger(123); % finishing stimuli
    Screen('TextSize', w, ceil(10));
    Screen('DrawText', w, '+', x1(1)+s_size/2 - textbox(3)/2,...
        y1(1)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(2)+s_size/2 - textbox(3)/2,...
        y1(2)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(3)+s_size/2 - textbox(3)/2,...
        y1(3)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    Screen('DrawText', w, '+', x1(4)+s_size/2 - textbox(3)/2,...
        y1(4)+s_size/2 - textbox(4)/2, [128, 128, 128]);
    tic; % getting wait time for network
    Screen('Flip', w);
    if opt.online
        flushinput(sock);
        result = fread(sock,1);
        if isempty(result)
            output = 'Sorry...';
            result = 1;
        end
        res = toc;
        Screen('FillRect', w, [255 0 0], [x1(result),y1(result),x2(result),y2(result)]);
        Screen('TextSize', w, ceil(10));
        Screen('DrawText', w, '+', x1(1)+s_size/2 - textbox(3)/2,...
            y1(1)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawText', w, '+', x1(2)+s_size/2 - textbox(3)/2,...
            y1(2)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawText', w, '+', x1(3)+s_size/2 - textbox(3)/2,...
            y1(3)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('DrawText', w, '+', x1(4)+s_size/2 - textbox(3)/2,...
            y1(4)+s_size/2 - textbox(4)/2, [128, 128, 128]);
        Screen('Flip', w);
    end
    tic; %wait for escapekey or waitkey
    while toc < opt.time_rest - res
        [ keyIsDown, ~, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(1);
                Screen('CloseAll');
                %                 ppTrigger(222);
                ppWrite(IO_ADD,222);
                fclose('all');
                return;
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'stop experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                GetClicks(w);
            end
        end
    end
end

%% Rest
Screen('TextSize',w, 50);
% DrawFormattedText(w,'Closed your eyes\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
% Screen('Flip', w);
% GetClicks(w);
% ppWrite(IO_ADD,77);
% Screen('Flip', w);
% WaitSecs(opt.rs_time);
% ppWrite(IO_ADD, 14);
DrawFormattedText(w,'Recording Resting state\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,78);
Screen('FillRect', w, [255 255 255], FixCross');
Screen('Flip', w);
WaitSecs(opt.rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'Thank you','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(2);

%% End
% Screen('TextSize',w, 50);
% DrawFormattedText(w, 'Thank you', 'center', 'center', [255 255 255]);
% Screen('Flip', w);
% WaitSecs(2);

% End trigger
ppWrite(IO_ADD,222);
% ppTrigger(222);

disp('Waiting for closing client socket');
WaitSecs(2);
if opt.online
    fclose(sock);
end
Screen('CloseAll');
ShowCursor;
fclose('all');
output = 'Finish';
Priority(0);

%% Waiting msg until n seconds

