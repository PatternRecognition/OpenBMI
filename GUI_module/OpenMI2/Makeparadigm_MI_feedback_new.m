function [ output_args ] = Makeparadigm_MI_feedback_new(varargin )

opt=opt_cellToStruct(varargin{:});
n1=0;n2=0;n3=0;
FOOTfb=3; % 이 부분.... 어찌해야되냐...

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

%% trigger setting
global IO_ADD IO_LIB;
IO_ADD=hex2dec(port);
IO_LIB=which('inpoutx64.dll');

%% server-client
% server eeg
t_e = tcpip('localhost', 3000, 'NetworkRole', 'Server');
set(t_e, 'InputBufferSize', 1024);
% Open connection to the client
fprintf('%s \n','Client Connecting...');
fopen(t_e)
fprintf('%s \n','Client Connected');
connectionServer = t_e;
set(connectionServer,'Timeout',.1);

%% image load
img_right=imread('right_mix.png');
img_right=imresize(img_right,2.5);
img_left=imread('left_mix.png');
img_left=imresize(img_left,2.5);
img_down=imread('down_mix.png');
img_down=imresize(img_down,2.5);
% cross=imread('cross.jpg');

%% order of stimulus (random)
sti_stack=[];

for i=1:length(type_sti)
    sti_stack(i,1:num_trial)=i;
end
for ii=1:size(sti_stack,2)
    sti_stack(:,ii)=(sti_stack(:,ii).*type_sti')';
end
idd=find(sti_stack);
sti_stack=reshape(sti_stack(idd),[num_trial,num_class])';
sti_stack=Shuffle(sti_stack(:));
%% jittering
a=-1;b=1;
% association with + time jittering
jitter = a + (b-a).*rand(num_trial*num_class,1);
jitter=jitter*0.5;

%% key setting
escapeKey = KbName('esc');
waitKey=KbName('*');

%% screen setting (gray)
% screenRes = [0 0 640 480];
if ischar(size_screen)
    screen_size='full';
else
    screen_size=[0, 0, size_screen];
end
black=BlackIndex(screenNumber);
if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, black);
else
    [w, wRect]=Screen('OpenWindow',screenNumber, black,screen_size);
end

% screens=Screen('Screens');
% screenNumber=max(screens);
% gray=GrayIndex(screenNumber);
% [w, wRect]=Screen('OpenWindow',screenNumber, gray,[0 0 500 500]);
[X,Y] = RectCenter(wRect);

%% fixation cross
CrossSize = 20;
FixCross = [X-1,Y-CrossSize,X+1,Y+CrossSize;X-CrossSize,Y-1,X+CrossSize,Y+1];
% screen size 변화에 따른, 화살표 size 변화... 를 한건데... 음...
% ssize=[X-size_screen(1)/4,Y-size_screen(1)/5,X+size_screen(1)/4,Y+size_screen(1)/5];

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
if ischar(size_screen)
    Textsize=50;
else
    Textsize=ceil(sqrt(size_screen(1)*size_screen(2))/22);
end
Screen('TextSize',w, Textsize);
% Screen('TextSize',w, 50);
DrawFormattedText(w,'Mouse click to start MI experiment \n\n (Press s to pause, esc to stop)','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,111)

% %% Eyes open/closed
% eyesOpenClosed2([0 0 0]) % script

%% paradigm start
Screen('Flip', w);
ppWrite(IO_ADD,111);
WaitSecs(time_blank);

for num_stimulus=1:length(sti_stack)

    if num_stimulus==length(sti_stack)/2 && length(sti_stack)>30%75 %~rem(num_stimulus,90) % 30회마다 pause
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
    
    start=GetSecs;
    while GetSecs < start+time_blank+jitter(num_stimulus)
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(1);
                Screen('CloseAll');
                ppWrite(IO_ADD,222);
                fclose('all');
                return
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'stop experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                GetClicks(w);
                break
            end
        end
    end
    
    Screen('FillRect', w, [255 255 255], FixCross');
    Screen('Flip', w);
    if sound
        Snd('Play',beep);
    end
    ppWrite(IO_ADD,15);
    WaitSecs(time_cross);
    switch sti_stack(num_stimulus)
        case 1
            n1=n1+1;
            ppWrite(IO_ADD,sti_stack(num_stimulus));
            image=img_right;
        case 2
            n2=n2+1;
            ppWrite(IO_ADD,sti_stack(num_stimulus));
            image=img_left;
        case 3
            n3=n3+1;
            ppWrite(IO_ADD,sti_stack(num_stimulus));
            image=img_down;
    end
    
    tex=Screen('MakeTexture', w, image );
    Screen('DrawTexture', w, tex);
    FixCross2=FixCross;
    Screen('FillRect', w, [255 0 0], FixCross2');
    [VBLTimestamp startrt]=Screen('Flip', w);
    start=GetSecs;
    
    while GetSecs < start+time_sti
        flushinput(t_e)
        cfout=fread(t_e,3, 'double')'
        
        if numel(cfout)==0
            disp('numel(cfout)==0 ...?')
            f_eeg=[];
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
                FixCross2(1,1)=FixCross2(1,1)-f_eeg;
                FixCross2(2,1)=FixCross2(2,1)-f_eeg;
                FixCross2(1,3)=FixCross2(1,3)-f_eeg;
                FixCross2(2,3)=FixCross2(2,3)-f_eeg;
            else
                FixCross2(1,2)=FixCross2(1,2)+f_eeg;
                FixCross2(2,2)=FixCross2(2,2)+f_eeg;
                FixCross2(1,4)=FixCross2(1,4)+f_eeg;
                FixCross2(2,4)=FixCross2(2,4)+f_eeg;
            end
        end
        
    end

    Screen('Flip', w);
    WaitSecs(time_blank);

    disp(sprintf('Trial #%.d',num_stimulus));
    disp(sprintf('1: %.0f\t2: %.0f\t3: %.0f',n1,n2,n3))

end


ppWrite(IO_ADD,222);
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [255 255 255]);
ShowCursor;
Screen('Flip', w);
WaitSecs(1);
Screen('CloseAll');
fclose('all');
Priority(0);

end

