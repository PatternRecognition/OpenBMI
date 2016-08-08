function [ output_args ] = paradigm_MI( opt )
%PARADIGM_MI Summary of this function goes here
%   Detailed explanation goes here
opt=opt_proplistToStruct_lower(opt{:});
global BMI
%% Default parameter setting
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

%% screen setting
screens=Screen('Screens');
if ~isfield(opt,'num_screen')
    screenNumber=max(screens);
else
    screenNumber=getfield(opt,'num_screen');
end

if ~isfield(opt,'screen_size')
    screen_size='full';
else
    screen_size=getfield(opt,'screen_size');
end

if ~isfield(opt,'screen_type')
    screen_size='window';
else
    screen_size=getfield(opt,'screen_size');
end
%%
if ~isfield(opt,'beep')
    beep='on';
else
    beep=getfield(opt,'beep');
end

% start matlab with administrator
% copy inpout32 or inpout64.dll file to window\system32 folder
%In this paradigm, trigger number is assigned as below:
%rightg=1, left=2, paradigm start=4, paradigm end=5
config_io;
which('right.jpg')
%image load
img_right=imread([BMI.PARADIGM_DIR '\MotorImagery\Stimulus\right.jpg']);
img_left=imread([BMI.PARADIGM_DIR '\MotorImagery\Stimulus\left.jpg']);
img_cross=imread([BMI.PARADIGM_DIR '\MotorImagery\Stimulus\cross.jpg']);

%Binary class
RepeatTimes=1;
num_class=2;

for i=1:num_class
    a1(i,1:num_trial)=i;
end

%stimulus will be presented randomly
[t s]=size(a1);
sti_stack=reshape(a1',1,t*s);
sti_stack=Shuffle(sti_stack);

% association with + time jittering
jitter = time_jitter + (1.3-1).*rand(num_trial*num_class,1);

% beep sound
if strcmp(beep,'on')
    [beep,samplingRate] = MakeBeep(10,100,[300]);
    Snd('Open');
    sound=1;
else
    sound=0;
end

gray=GrayIndex(screenNumber);
if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, gray);
else
    [w, wRect]=Screen('OpenWindow',screenNumber, gray,[10 10 screen_size]);
end

% Wait for mouse click:
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Press X, S to quit or the stop paradigm when cross appear \\ Mouse click to start NIRS experiment', 'center', 'center', [0 0 0]);
Screen('Flip', w);
GetClicks(w);

escapeKey = KbName('esc');
waitKey=KbName('s');

p_close=0;
for i=1:RepeatTimes
    for num_stimulus=1:length(sti_stack)
        send_ppTrigger(5);
        tex=Screen('MakeTexture', w, img_cross);
        Screen('DrawTexture', w, tex);
        Screen('Close',tex)
        [VBLTimestamp startrt]=Screen('Flip', w);
        
        start=GetSecs;
        while GetSecs < start+time_isi+jitter(num_stimulus)-2
            [ keyIsDown, seconds, keyCode ] = KbCheck;
            if keyIsDown
                if keyCode(escapeKey)
                    ShowCursor;
                    p_close=1;                    
                    break;
                elseif keyCode(waitKey)
                    warning('stop')
                    tex=Screen('MakeTexture', w, img_cross);
                    Screen('DrawTexture', w, tex);
                    GetClicks(w);
                    Screen('Close',tex);
                    [VBLTimestamp startrt]=Screen('Flip', w);
                end
            end
            pause(0.1);
        end
        
        if p_close
            break;
        end
        
        if sound
            Snd('Play',beep);
        end
        WaitSecs(2);
        
        switch sti_stack(num_stimulus)
            case 1 %right class
                send_ppTrigger(1);
                image=img_right;
                tex=Screen('MakeTexture', w, image );
                Screen('DrawTexture', w, tex);
                [VBLTimestamp startrt]=Screen('Flip', w);
                
                WaitSecs(time_sti);
                Screen('Close',tex);
                
            case 2 %left class
                send_ppTrigger(2);
                image=img_left;
                tex=Screen('MakeTexture', w, image );
                Screen('DrawTexture', w, tex);
                [VBLTimestamp startrt]=Screen('Flip', w);
                
                WaitSecs(time_sti);
                Screen('Close',tex);
        end
        num_stimulus
    end
end

send_ppTrigger(6);
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Thank you.', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(2);

Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);
end
