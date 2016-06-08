function [ output_args ] = Makeparadigm_MI( opt )

opt=opt_proplistToStruct_lower(opt{:});
global BMI

if ~isfield(opt,'time_sti')
    time_sti=5;
else
    time_sti=getfield(opt,'time_sti');
end

if ~isfield(opt,'time_rest')
    time_rest=5;
else
    time_rest=getfield(opt,'time_rest');
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

if ~isfield(opt,'num_class')
    num_class=2;
else
    num_class=getfield(opt,'num_class');
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
    screen_type='window';
else
    screen_type=getfield(opt,'screen_type');
end


%% beep setting
beep='on';
beep_time=2;


%image load
img_right=imread('Stimulus\right.jpg');
img_left=imread('Stimulus\left.jpg');
img_down=imread('Stimulus\down.jpg');
img_cross=imread('Stimulus\cross.jpg');


for i=1:num_class
    a1(i,1:num_trial)=i;
end

%stimulus will be presented randomly
[t s]=size(a1);
sti_stack=reshape(a1',1,t*s);
sti_stack=Shuffle(sti_stack);

a=-1;b=1;

% association with + time jittering
jitter = a + (b-a).*rand(100,1);
jitter=jitter*0.5;

% beep sound
if strcmp(beep,'on')
    [beep,samplingRate] = MakeBeep(10,100,[300]);
    Snd('Open');
    sound=1;
else
    sound=0;
end

gray=GrayIndex(2);

if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, gray);
else
    [w, wRect]=Screen('OpenWindow', 2, gray);
end

% Wait for mouse click:
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Press X, S to quit or the stop paradigm when cross appear \\ Mouse click to start MI experiment', 'center', 'center', [0 0 0]);
Screen('Flip', w);
GetClicks(w);

escapeKey = KbName('esc');
waitKey=KbName('s');

p_close=0;

%% make fixation cross
fixSize =80;
[X,Y] = RectCenter(wRect);
PixofFixationSize = 80;
FixCross = [X-1,Y-PixofFixationSize,X+1,Y+PixofFixationSize;X-PixofFixationSize,Y-1,X+PixofFixationSize,Y+1];

%% paradigm start
for num_stimulus=1:length(sti_stack)
%         send_ppTrigger(5);
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    WaitSecs(time_isi-1)
    
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
                GetClicks(w);
                Screen('Close',tex1);
                
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
        
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    WaitSecs(time_rest)
    
    switch sti_stack(num_stimulus)
        case 1 %right class
            %send_ppTrigger(1);
            image=img_right;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            [VBLTimestamp startrt]=Screen('Flip', w);
            
            WaitSecs(time_sti);
            Screen('Close',tex1);
            
        case 2 %left class
            %                 send_ppTrigger(2);
            image=img_left;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            [VBLTimestamp startrt]=Screen('Flip', w);
            
            WaitSecs(time_sti);
            Screen('Close',tex1);
            
        case 3 %foot class
            %                 send_ppTrigger(3);
            image=img_down;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            [VBLTimestamp startrt]=Screen('Flip', w);
            
            WaitSecs(time_sti);
            Screen('Close',tex1);
    end
    num_stimulus
end


% send_ppTrigger(6);
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Thank you.', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(2);
Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);

