function [ output_args ] = Makeparadigm_MI( varargin )
% Makeparadigm_MI (Experimental paradigm):
% 
% Description:
%   Basic motor imagery experiment paradigm using psychtoolbox.
%   It shows a cross, an arrow, and blank screen alternately.
% 
% Example:
%   Makeparadigm_MI({'time_cross',1.5;'time_sti',60;'time_blank',5;'num_trial',1;'num_class',3});
% 
% Input: (Nx2 size, cell-type)
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
% screenNumber=2;

%% screen setting
screens=Screen('Screens');
if ~isfield(opt,'num_screen'),screenNumber=max(screens); else screenNumber=opt.num_screen; end
if ~isfield(opt,'screen_size'),screen_size='full'; else screen_size=opt.screen_size; end
% if ~isfield(opt,'screen_type'),screen_type='window'; else screen_type=opt.screen_type; end

%% beep setting
beep='on';
freq=22000;
beepLengthSecs=0.5;

%% trigger setting
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('D010');
IO_LIB=which('inpoutx64.dll');

%% jittering
a=-1;b=1;
% association with + time jittering
jitter = a + (b-a).*rand(num_trial*num_class,1);
jitter=jitter*0.5;

%% image load
currentFile = mfilename( 'fullpath' );
[pathstr,~,~] = fileparts( currentFile );
img_right=imread(fullfile(pathstr, "..", '\Stimulus\right.jpg'));
img_left=imread(fullfile(pathstr, "..",'\Stimulus\\left.jpg'));
img_down=imread(fullfile(pathstr, "..",'\Stimulus\down.jpg'));
% img_cross=imread('\Stimulus\cross.jpg');

%% beep sound
if strcmp(beep,'on')
    [beep,samplingRate] = MakeBeep(500,beepLengthSecs,freq);
    Snd('Open');
    sound=1;
else
    sound=0;
end
% Screen('Preference', 'SkipSyncTests', 1);

%% psychtoolbox setting
gray=GrayIndex(screenNumber);
% screenRes = [0 0 300 300];
if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, gray);
else
    [w, wRect]=Screen('OpenWindow',screenNumber, gray, screen_size);
end

%% order of stimulus (random)
for i=1:num_class
    a1(i,1:num_trial)=i;
end
% a1=Shuffle(a1);
% [t s]=size(a1);
% sti_stack=reshape(a1,1,t*s);
sti_stack=Shuffle(a1(:)'); % smkim


% click to start:
Screen('TextSize',w, 50);
DrawFormattedText(w,'Mouse click to start MI experiment \n\n (Press s to pause, esc to stop)','center','center',[0 0 0]);
Screen('Flip', w);
GetClicks(w);
ppTrigger(111); % START

escapeKey = KbName('esc');
waitKey=KbName('s');
p_close=0;

%% Eyes open/closed
eyesOpenClosed2 % script

%% fixation cross
[X,Y] = RectCenter(wRect);
FixationSize = 20;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% paradigm start
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    WaitSecs(2.5)

for num_stimulus=1:length(sti_stack)
    
    if num_stimulus==length(sti_stack)/2
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
    while GetSecs < start+time_blank+jitter(num_stimulus)-2
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
%                 ShowCursor;
%                 p_close=1;
                Screen('CloseAll');
                fclose('all');
                return
            elseif keyCode(waitKey)
                warning('stop')
                GetClicks(w);
                Screen('Close',tex1);
                
            end
        end
        pause(0.1);
    end
    
%     if p_close
%         break;
%     end
    
    Screen('FillRect', w, [0 0 0], FixCross');
    Screen('Flip', w);
    if sound
        Snd('Play',beep);
    end
    
    WaitSecs(1)
    ppTrigger(5);
    WaitSecs(time_cross)

    switch sti_stack(num_stimulus)
        case 1 % right class
            ppTrigger(1);
            image=img_right;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            Screen('Flip', w);
            
            WaitSecs(time_sti);
            Screen('Close',tex1);
%             ppTrigger(11);
            
        case 2 % left class
            ppTrigger(2);
            image=img_left;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            Screen('Flip', w);

            WaitSecs(time_sti);
            Screen('Close',tex1);
%             ppTrigger(22);
            
        case 3 % foot class
            ppTrigger(3);
            image=img_down;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            Screen('Flip', w);
            
            WaitSecs(time_sti);
            Screen('Close',tex1);
%             ppTrigger(33);
    end
    num_stimulus
    
    Screen('Flip', w);
    WaitSecs(time_blank)

end

ppTrigger(222);
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [0 0 0]);
Screen('Flip', w);
WaitSecs(2);
Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);

