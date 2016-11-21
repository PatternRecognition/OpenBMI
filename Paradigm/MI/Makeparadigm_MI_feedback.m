function [ output_args ] = Makeparadigm_MI_feedback(weight, bias, FOOTfb, varargin )
% Makeparadigm_MI_feedback (Experimental paradigm):
% 
% Description:
%   Basic motor imagery experiment paradigm using psychtoolbox.
%   It shows a cross, an arrow, and blank screen alternately.
%   As a feedback, red cross at center moves according to classified
%   results in real time.
% 
% Example:
%   Makeparadigm_MI_feedback(weight, bias, FOOTfb, {'time_sti',4;'time_blank',3;'time_cross',3;'num_trial',50;'num_class',3});
% 
% Input:
%   weight - 1x2 vector
%   bias   - 1x2 vector
%   FOOTfb - 2 or 3
% Option: (Nx2 size, cell-type)
%   time_cross  - time for concentration, showing a cross [s]
%   time_sti    - time for a stimulus, showing an arrow (right, left, or down) [s]
%   time_blank  - time for rest, showing nothing but gray screen [s]
%   num_trial   - number of trials per class
%   num_class   - number of class you want, 1 to 3, (right, left, and foot)
%   num_screen  - number of screen to show 
%   screen_size - size of window showing experimental stimulus
%                 'full' or matrix (e.g.[0 0 300 300])
% 

opt=opt_cellToStruct(varargin{:});

%% default setting
if ~isfield(opt,'time_sti'),    time_sti=4;      else time_sti=opt.time_sti;      end
if ~isfield(opt,'time_cross'),  time_cross=3;    else time_cross=opt.time_cross;  end
if ~isfield(opt,'time_blank'),  time_blank=3;    else time_blank=opt.time_blank;  end
if ~isfield(opt,'num_trial'),   num_trial=50;    else num_trial=opt.num_trial;    end
% if ~isfield(opt,'time_jitter'), time_jitter=0.1; else time_jitter=opt.time_jitter;end
if ~isfield(opt,'num_class'),   num_class=3;     else num_class=opt.num_class;    end

%% trigger setting
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('D010');
IO_LIB=which('inpoutx64.dll');

%% server-client
% server eeg
t_e = tcpip('localhost', 3000, 'NetworkRole', 'Server');
set(t_e, 'InputBufferSize', 3000);
% Open connection to the client
fopen(t_e);
fprintf('%s \n','Client Connected');
connectionServer = t_e;
set(connectionServer,'Timeout',.1);

%% image load
img_right=imread('right.jpg');
img_left=imread('left.jpg');
img_down=imread('down.jpg');
cross=imread('cross.jpg');

%% order of stimulus (random)
sti_stack=[];
for i=1:num_class
    tp(1:num_trial)=i;
    sti_stack=cat(2,sti_stack,tp);
end
sti_stack=Shuffle(sti_stack);

%% jittering
a=-1;b=1;
% association with + time jittering
jitter = a + (b-a).*rand(num_trial*num_class,1);
jitter=jitter*0.5;

escapeKey = KbName('esc');
waitKey=KbName('s');

%% screen setting (gray)
% screenRes = [0 0 640 480];
screens=Screen('Screens');
if ~isfield(opt,'num_screen'),screenNumber=max(screens); else screenNumber=opt.num_screen; end
if ~isfield(opt,'screen_size'),screen_size='full'; else screen_size=opt.screen_size; end
gray=GrayIndex(screenNumber);
if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, gray);
else
    [w, wRect]=Screen('OpenWindow',screenNumber, gray, screen_size);
end
[X,Y] = RectCenter(wRect);

%% fixation cross
PixofFixationSize = 20;
FixCross = [X-1,Y-PixofFixationSize,X+1,Y+PixofFixationSize;X-PixofFixationSize,Y-1,X+PixofFixationSize,Y+1];

%% beep setting
beep='on';
freq=22000;
beepLengthSecs=0.5;

%% beep sound
if strcmp(beep,'on')
    [beep,samplingRate] = MakeBeep(500,beepLengthSecs,freq);
    Snd('Open');
    sound=1;
else
    sound=0;
end

% click to start:
Screen('TextSize',w, 50);
DrawFormattedText(w,'Mouse click to start MI experiment \n\n (Press s to pause, esc to stop)','center','center',[0 0 0]);
Screen('Flip', w);
GetClicks(w);
ppTrigger(111)

%% Eyes open/closed
eyesOpenClosed2 % script

%% paradigm start
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    WaitSecs(2.5)

exp_on=true;

for num_stimulus=1:length(sti_stack)
    
    if num_stimulus==length(sti_stack)/2 %75 %~rem(num_stimulus,90) % 30È¸¸¶´Ù pause
        Screen('TextSize',w, 50);
        DrawFormattedText(w,'Rest\n\n(Pause the brain vision)','center','center',[0 0 0]);
        Screen('Flip', w);
        GetClicks(w);
        DrawFormattedText(w,'(Resume recording)','center','center',[0 0 0]);
        Screen('Flip', w);
        GetClicks(w);
        DrawFormattedText(w,'Click to continue the experiment','center','center',[0 0 0]);
        Screen('Flip', w);
        GetClicks(w);
    end
    start=GetSecs;
    while GetSecs < start+time_blank+jitter(num_stimulus)
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
%                 tex=Screen('MakeTexture', w, cross);
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
%                 exp_on=false;
                WaitSecs(1);
                Screen('CloseAll');
                fclose('all');
                return
%                 break
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'Mouse click to restart an experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                GetClicks(w);
                
                tex=Screen('MakeTexture', w, cross);
                Screen('Flip', w);
                WaitSecs(time_blank+jitter(num_stimulus));
                break;
            end
        end
    end
    
    % esc key in the cross state
    if ~exp_on
        break;
    end
    
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    if sound
        Snd('Play',beep);
    end
    
    WaitSecs(1)
    ppTrigger(5);
    WaitSecs(time_cross)
    
    switch sti_stack(num_stimulus)
        case 1
            ppTrigger(sti_stack(num_stimulus))
            image=img_right;
        case 2
            ppTrigger(sti_stack(num_stimulus))
            image=img_left;
        case 3
            ppTrigger(sti_stack(num_stimulus))
            image=img_down;
    end
    tex=Screen('MakeTexture', w, image );
    Screen('DrawTexture', w, tex);
    FixCross2=FixCross;
    Screen('FillRect', w, [255 0 0], FixCross2');
    [VBLTimestamp startrt]=Screen('Flip', w);
    start=GetSecs;
    while GetSecs < start+time_sti
%         tic
        cfout=fread(t_e,3, 'double');
        if numel(cfout)==0
            disp('cfout==0 ...?')
        else
        switch sti_stack(num_stimulus)
            case 1
                f_eeg=cfout(1);
            case 2
                f_eeg=cfout(1);
            case 3 % select a clf output data for foot motor imagery
                f_eeg=cfout(FOOTfb);
        end
        end
        if ~isempty(f_eeg)
            tex=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex);
            Screen('FillRect', w, [255 0 0], FixCross2');
            [VBLTimestamp startrt]=Screen('Flip', w);
            if sti_stack(num_stimulus)==1 || sti_stack(num_stimulus)==2
                FixCross2(1,1)=FixCross2(1,1)-(f_eeg*weight(1))+bias(1);
                FixCross2(2,1)=FixCross2(2,1)-(f_eeg*weight(1))+bias(1);
                FixCross2(1,3)=FixCross2(1,3)-(f_eeg*weight(1))+bias(1);
                FixCross2(2,3)=FixCross2(2,3)-(f_eeg*weight(1))+bias(1);
            else
                FixCross2(1,2)=FixCross2(1,2)-(f_eeg*weight(2))-bias(2);
                FixCross2(2,2)=FixCross2(2,2)-(f_eeg*weight(2))-bias(2);
                FixCross2(1,4)=FixCross2(1,4)-(f_eeg*weight(2))-bias(2);
                FixCross2(2,4)=FixCross2(2,4)-(f_eeg*weight(2))-bias(2);
            end
        end
    end
    disp(sprintf('Trial #%.d',num_stimulus));
    
    Screen('Flip', w);
    WaitSecs(time_blank)
    
end


ppTrigger(222) % END
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [0 0 0]);
ShowCursor;
Screen('Flip', w);
WaitSecs(1);
Screen('CloseAll');
fclose('all');
Priority(0);


end

