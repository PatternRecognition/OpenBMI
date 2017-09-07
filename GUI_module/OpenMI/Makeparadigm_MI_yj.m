function [ output_args ] = Makeparadigm_MI_yj(varargin)

opt=opt_cellToStruct(varargin{:});
n1=0;n2=0;n3=0;

%% default setting
if isempty(opt.time_sti), time_sti=3; else time_sti=opt.time_sti; end
if isempty(opt.num_trial), num_trial=150; else num_trial=opt.num_trial; end
if isempty(opt.num_class), num_class=3; else num_class=opt.num_class; end
if isempty(opt.num_screen), screenNumber=2; else screenNumber=opt.num_screen; end
if isempty(opt.size_screen), size_screen='full'; else size_screen=opt.size_screen; end
if ~isfield(opt,'time_cross'),  time_cross=3;    else time_cross=opt.time_cross;  end
if ~isfield(opt,'time_blank'),  time_blank=3;    else time_blank=opt.time_blank;  end
if ~isfield(opt,'port'), error('No input port information'); else port= opt.port;end
if ~isfield(opt,'eyes'), opt.eyes=0; end
type_sti=opt.type_sti;

%% screen setting
screens=Screen('Screens');
if ~isfield(opt,'num_screen'),screenNumber=max(screens);end
if ~isfield(opt,'screen_size'),screen_size='full';end
if ~isfield(opt,'screen_type'),screen_type='window';end

%% beep setting
beep='on';
freq=22000;
beepLengthSecs=0.5;

%% trigger setting
global IO_ADD IO_LIB;
IO_ADD=hex2dec(port);
IO_LIB=which('inpoutx64.dll');

%% jittering
a=-1;b=1;
% association with + time jittering
jitter = a + (b-a).*rand(num_trial*num_class,1);
jitter=jitter*0.5;

%% image load
img_right=imread('right_mix.png');
img_right=imresize(img_right,2.5);
img_left=imread('left_mix.png');
img_left=imresize(img_left,2.5);
img_down=imread('down_mix.png');
img_down=imresize(img_down,2.5);

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
% Screen('Preference', 'SkipSyncTests', 1);
if ischar(size_screen)
    screen_size='full';
else
    screen_size=[0, 0, size_screen];
end
black = BlackIndex(screenNumber);
% screenRes = [0 0 300 300];
if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, black);
else
    [w, wRect]=Screen('OpenWindow',screenNumber, black, screen_size);
end

%% order of stimulus (random)
for i=1:length(type_sti)
    a1(i,1:num_trial)=i;
end
for ii=1:num_trial % 클래스 개수 . 여기 포문 없앨수 수 있을거가튼뎀?
a1(:,ii)=(a1(:,ii).*type_sti')';
end
idd=find(a1);
a1=reshape(a1(idd),[num_trial,num_class])';
% for i=1:num_class
%     a1(i,1:num_trial)=i;
% end
% if num_class==1
    
% a1=Shuffle(a1);
% [t s]=size(a1);
% sti_stack=reshape(a1,1,t*s);
sti_stack=Shuffle(a1(:)'); % smkim


% click to start:
if ischar(size_screen)
    Textsize=50;
else
    Textsize=ceil(sqrt(size_screen(1)*size_screen(2))/22);
end

Screen('TextSize',w, Textsize);
DrawFormattedText(w,'Mouse click to start MI experiment \n\n (Press s to pause, esc to stop)','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,111);

escapeKey = KbName('esc');
waitKey=KbName('*');
% p_close=0;

%% Eyes open/closed
% if opt.eyes
%     textColor=[0 0 0];
% eyesOpenClosed2 % script
% end

%% fixation cross
[X,Y] = RectCenter(wRect);
FixationSize = 20;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];
% screen size 변화에 따른, 화살표 size 변화... 를 한건데... 음...
% if ischar(size_screen)
%     ssize=[X-200,Y-200,X+200,Y+200];
% else
%     ssize=[X-size_screen(1)/4,Y-size_screen(1)/5,X+size_screen(1)/4,Y+size_screen(1)/5];
% end

%% paradigm start
    Screen('Flip', w);
    ppWrite(IO_ADD,111);
    WaitSecs(time_blank);

for num_stimulus=1:length(sti_stack)
%     if num_stimulus==50 || num_stimulus==100
    
    % Pause 실험 반 하고 좀 쉬어야지
    if num_stimulus==length(sti_stack)/2 && length(sti_stack)>50

        Screen('TextSize',w, Textsize);
        DrawFormattedText(w,'Rest\n\n(Pause the brain vision)','center','center',[255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
        DrawFormattedText(w,'(Resume recording)','center','center',[255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
        DrawFormattedText(w,'Click to continue the experiment','center','center',[255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
        WaitSecs(2.5);
    end
    
    start=GetSecs;
    while GetSecs < start+time_blank+jitter(num_stimulus)-2
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(1);
                Screen('CloseAll');
%                 ppTrigger(222);
                ppWrite(IO_ADD,222);
                fclose('all');
                return
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'stop experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                GetClicks(w);
            end
        end
        pause(0.1);
    end

    
    Screen('FillRect', w, [255 255 255], FixCross');
    Screen('Flip', w);
    if sound
        Snd('Play',beep);
    end
    ppWrite(IO_ADD,15);
    WaitSecs(time_cross);
    
    switch sti_stack(num_stimulus)
        case 1 % right class
            n1=n1+1;
            ppWrite(IO_ADD,1);
            image=img_right;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            Screen('Flip', w);
            WaitSecs(time_sti);

        case 2 % left class
            n2=n2+1;
            ppWrite(IO_ADD,2);
            image=img_left;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            Screen('Flip', w);
            WaitSecs(time_sti);
            
        case 3 % foot class
            n3=n3+1;
            ppWrite(IO_ADD,3);
            image=img_down;
            tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);
            Screen('FillRect', w, [0 0 0], FixCross');
            Screen('Flip', w);
            WaitSecs(time_sti);
    end
    
    Screen('Flip', w);
    WaitSecs(time_blank);
    
    disp(sprintf('Trial #%.d',num_stimulus));
    disp(sprintf('1: %.0f\t2: %.0f\t3: %.0f',n1,n2,n3))
    
end
ppWrite(IO_ADD,222);
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [255 255 255]);
Screen('Flip', w);
WaitSecs(2);
Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);

