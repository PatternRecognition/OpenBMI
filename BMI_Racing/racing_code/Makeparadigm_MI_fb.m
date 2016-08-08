function [ output_args ] = Makeparadigm_MI_fb( varargin )

global BMI
opt=opt_cellToStruct(varargin{:});

if ~isfield(opt,'time_sti')
    time_sti=5;
else
    time_sti=getfield(opt,'time_sti');
end
if ~isfield(opt,'time_isi')
    time_isi=8;
else
    time_isi=getfield(opt,'time_isi');
end

if ~isfield(opt,'num_trial')
    num_trial=30;
else
    num_trial=getfield(opt,'num_trial');
end

if ~isfield(opt,'time_jitter')
    time_jitter=0.1;
else
    time_jitter=getfield(opt,'time_jitter');
end

if ~isfield(opt,'classes')
    classes={'right','left','foot'};
    num_class=length(classes);
else
    classes=getfield(opt,'classes');
    num_class=length(classes);
end

%% screen setting
screens=Screen('Screens');

if ~isfield(opt,'screen')
    screen_size='full';
else
    screen_size=getfield(opt,'screen');
end


%% Server
TCPIP=false;
if isfield(opt, 'TCPIP')
    if strcmp(opt.TCPIP,'on')
        %server eeg
        t_e = tcpip('localhost', 3000, 'NetworkRole', 'Server');
        set(t_e , 'InputBufferSize', 3000);
        % Open connection to the client.
        fopen(t_e);
        fprintf('%s \n','Client Connected');
        connectionServer = t_e;
        set(connectionServer,'Timeout',0);
        TCPIP=true;
    end
end

%% beep setting
beep='on';
beep_time=2;

%% trigger setting
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('C010');
IO_LIB=which('inpoutx64.dll');

%% image load
img_right=imread('\Stimulus\right_arrow.jpg');
img_left=imread('\Stimulus\\left_arrow.jpg');
img_foot=imread('\Stimulus\up_arrow.jpg');
img_rest=imread('\Stimulus\rest_square.jpg');


for i=1:3  % 3 is maximum number of active classes, the rest class is being appears before active stimulus, the resting class is not include here, it appears before active classes.
    a1(i,1:num_trial)=i;
end
a1=Shuffle(a1);

%stimulus will be presented randomly
[t s]=size(a1);
sti_stack=reshape(a1,1,t*s);
% set_classes={1, 'right'; 2, 'left'; 3, 'foot'} % 1, numbers of trigger, 2, class
% for i=1:length(set_classes)
%     if ~ismember(classes, set_classes{i})
%         [a b]=find(sti_stack~=i);
%         sti_stack=sti_stack(b);
%     end
% end
% ismember(set_classes{:,2},classes{i})
% strcmp(classes{1})

a=-1;b=1;

% association with + time jittering
jitter = a + (b-a).*rand(num_trial*num_class,1);
jitter=jitter*0.5;

% beep sound
if strcmp(beep,'on')
    [beep,samplingRate] = MakeBeep(10,100,[300]);
    Snd('Open');
    sound=1;
else
    sound=0;
end

screenNumber=2;
gray=GrayIndex(screenNumber);
if strcmp(screen_size, 'full')
    [w, wRect]=Screen('OpenWindow', 2, gray);
else
    screenRes = [0 0 640 480];
    [w, wRect]=Screen('OpenWindow',screenNumber, gray, screenRes);
end

%% Open an on screen window
% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(wRect);
% text initialize
outpXpos = xCenter;
outpYpos = yCenter - 30;
%-----------------------------------------------------------------
ppTrigger(111);
% Wait for mouse click:
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Press X, S to quit or the stop paradigm when cross appear \\ Mouse click to start MI experiment', 'center', 'center', [0 0 0]);
Screen('Flip', w);
GetClicks(w);

escapeKey = KbName('esc');
waitKey=KbName('s');
continueKey=KbName('c');

p_close=0;

%% make fixation cross
fixSize =20;
[X,Y] = RectCenter(wRect);
PixofFixationSize = 10;
FixCross = [X-1,Y-PixofFixationSize,X+1,Y+PixofFixationSize;X-PixofFixationSize,Y-1,X+PixofFixationSize,Y+1];

cly_update=[15 30 60 90 120 150 180];
feedback_on=false;
sound_once=true;
%% paradigm start
for num_stimulus=1:length(sti_stack)
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    %     WaitSecs(time_isi-1)
    start=GetSecs;
    while GetSecs < start+time_isi
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                ShowCursor;
                p_close=1;
                break;
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'Mouse click to start MI experiment', 'center', 'center', [0 0 0]);
                Screen('Flip', w);
                GetClicks(w);
                pause(1);
                GetClicks(w);
            else
                
            end
        end
        if GetSecs>start+time_isi-2   % 2s means the beep sound
            if sound && sound_once
                Snd('Play',beep);
                Screen('FillRect', w, [0 0 0], FixCross');
                Screen('Flip', w);
                sound_once=false;
            end
        end
        pause(0.05);
    end
    sound_once=true;
    if p_close
        break;
    end
    num_stimulus
    if sum(ismember(classes,'rest'))
        % resting class
        ppTrigger(4);
        image=img_rest;
        start=GetSecs;
        while GetSecs < start+time_sti
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            if TCPIP
                cfout=fread(t_e,4, 'double');
                if ~isempty(cfout) && feedback_on
                    c_str=cfout(1);
                    c_str = floor(c_str);
                    Screen('TextSize',w, 30);
                    if length(num2str(c_str))==1
                        t_str=[' ' num2str(c_str)];
                    else
                        t_str=num2str(c_str);
                    end
                    
%                     DrawFormattedText(w, t_str, outpXpos-10, outpYpos-15, [0 0 255]);
                end
            end
            [VBLTimestamp startrt]=Screen('Flip', w);
            %         pause(0.05);
        end
        ppTrigger(44);
        Screen('Close',tex1);
        Screen('FillRect', w, [0 0 0], FixCross');
        [VBLTimestamp startrt]=Screen('Flip', w);  % blank for a while
        pause(2);
    end
    switch sti_stack(num_stimulus)
        case 1 %right class
            ppTrigger(1);
            image=img_right;
        case 2 %left class
            ppTrigger(2);
            image=img_left;
        case 3 %foot class
            ppTrigger(3);
            image=img_foot;
    end
    start=GetSecs;
    while GetSecs < start+time_sti        
        tex1=Screen('MakeTexture', w, image );
        Screen('DrawTexture', w, tex1);
        Screen('FillRect', w, [0 0 0], FixCross');
        if TCPIP
            cfout=fread(t_e,4, 'double');
            if ~isempty(cfout) && feedback_on
                c_str=cfout(sti_stack(num_stimulus)+1);
                c_str =  floor(c_str);
                Screen('TextSize',w, 30);
                if length(num2str(c_str))==1
                    t_str=[' ' num2str(c_str)];
                else
                    t_str=num2str(c_str);
                end
                
%                 DrawFormattedText(w,t_str, outpXpos-10, outpYpos-15, [0 0 255]);
            end
        end
        [VBLTimestamp startrt]=Screen('Flip', w);
    end
    switch sti_stack(num_stimulus) % for end trigger, it needs for matlab2(client)
        case 1 %right class
            ppTrigger(11);
        case 2 %left class
            ppTrigger(22);
        case 3 %foot class
            ppTrigger(33);
    end
    if ismember(num_stimulus,cly_update) % initial pause time for classifier training
        
        if TCPIP
            tm1=true; tm2=false;
            while tm1
                DrawFormattedText(w, 'Clasifier training....', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                cfout=fread(t_e,1, 'double');
                if ~isempty(cfout)
                    if cfout==1  % TCP/IP 1 is for stop
                        tm2=true;
                    end
                end
                if tm2
                    while tm2
                        DrawFormattedText(w, 'Classifier constructed, press "C" for continue...', 'center', 'center', [0 0 0]);
                        Screen('Flip', w);
                        [ keyIsDown, seconds, keyCode ] = KbCheck;
                        if keyIsDown
                            if keyCode(continueKey)
                                tm1=false;
                                tm2=false;
                                feedback_on=true;
                            end
                        end
                    end
                end
            end
            DrawFormattedText(w, 'Mouse click to start MI experiment', 'center', 'center', [0 0 0]);
            Screen('Flip', w);
            GetClicks(w);
        else
            DrawFormattedText(w, 'Mouse click to start MI experiment', 'center', 'center', [0 0 0]);
            Screen('Flip', w);
            GetClicks(w);
            pause(1);
        end
        
    end
    
end
WaitSecs(1);

ppTrigger(6);
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Thank you.', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(2);
Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);

