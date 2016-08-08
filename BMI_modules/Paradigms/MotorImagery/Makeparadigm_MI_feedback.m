function [ output_args ] = Makeparadigm_MI_feedback(weight, bias, FOOTfb, varargin )
%FEEDBACK_SERVER Summary of this function goes here
%   Detailed explanation goes here
opt=opt_cellToStruct(varargin{:});

global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('D010');
IO_LIB=which('inpoutx64.dll');

%server eeg
t_e = tcpip('localhost', 3000, 'NetworkRole', 'Server');
set(t_e , 'InputBufferSize', 3000);
% Open connection to the client.
fopen(t_e);
fprintf('%s \n','Client Connected');
connectionServer = t_e;
set(connectionServer,'Timeout',.1);

beep='on';
beep_time=2;

% beep sound
if strcmp(beep,'on')
    [beep,samplingRate] = MakeBeep(10,100,[300]);
    Snd('Open');
    sound=1;
else
    sound=0;
end


%image load
M_RIGHT=imread('right.jpg');
M_LEFT=imread('left.jpg');
M_DOWN=imread('down.jpg');
cross=imread('cross.jpg');

nStimuli=opt.num_trial;
nClass=opt.num_class;
%% find
sti_stack=[];
for i=1:nClass
    tp(1:nStimuli)=i;
    sti_stack=cat(2,sti_stack,tp)
end
sti_stack=Shuffle(sti_stack);

%%
a=-1;b=1;
% association with + time jittering
jitter = a + (b-a).*rand(nStimuli*nClass,1);
jitter=jitter*0.5;

%interval setting
RepeatTimes=1;
sti_Times=opt.time_sti;%per/S 6
sti_Interval=opt.time_rest;%per/S 10

escapeKey = KbName('esc');
waitKey=KbName('s');

%screen setting (gray)

% screenRes = [0 0 640 480];
screens=Screen('Screens');
screenNumber=max(screens);
gray=GrayIndex(screenNumber);
% [w, wRect]=Screen('OpenWindow',2, gray,screenRes);
[w, wRect]=Screen('OpenWindow',2, gray);
fixSize = 10;
% ScreenP = Psychtoolbox_Open_Kb(screenNumber,fixSize);
[X,Y] = RectCenter(wRect);
PixofFixationSize = 20;
FixCross = [X-1,Y-PixofFixationSize,X+1,Y+PixofFixationSize;X-PixofFixationSize,Y-1,X+PixofFixationSize,Y+1];


% Wait for mouse click:
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Mouse click to start an experiment', 'center', 'center', [0 0 0]);
Screen('Flip', w);
GetClicks(w);
ppTrigger(5)


exp_on=true;
for i=1:RepeatTimes
    for num_stimulus=1:length(sti_stack)
        %         tex=Screen('MakeTexture', w, cross);
        Screen('FillRect', w, [0 0 0], FixCross');
        [VBLTimestamp startrt]=Screen('Flip', w);
        %         WaitSecs(sti_Interval+jitter(num_stimulus));
        
        start=GetSecs;
        while GetSecs < start+sti_Interval+jitter(num_stimulus)
            [ keyIsDown, seconds, keyCode ] = KbCheck;
            if keyIsDown
                if keyCode(escapeKey)
                    tex=Screen('MakeTexture', w, cross);
                    DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                    Screen('Flip', w);
                    exp_on=false;
                    WaitSecs(3);
                    break;
                elseif keyCode(waitKey)
                    DrawFormattedText(w, 'Mouse click to restart an experiment', 'center', 'center', [255 255 255]);
                    Screen('Flip', w);
                    GetClicks(w);
                    
                    tex=Screen('MakeTexture', w, cross);
                    Screen('Flip', w);
                    WaitSecs(sti_Interval+jitter(num_stimulus));
                    break;
                end
            end
        end
        
        % esc key in the cross state
        if ~exp_on
            break;
        end
        if sound
            Snd('Play',beep);
        end
            WaitSecs(1)
            
        switch sti_stack(num_stimulus)
            case 1
                ppTrigger(sti_stack(num_stimulus))
                image=M_RIGHT;
            case 2
                ppTrigger(sti_stack(num_stimulus))
                image=M_LEFT;
            case 3
                ppTrigger(sti_stack(num_stimulus))
                image=M_DOWN;
        end        
        tex=Screen('MakeTexture', w, image );
        Screen('DrawTexture', w, tex);
        FixCross2=FixCross;
        Screen('FillRect', w, [255 0 0], FixCross2');
        [VBLTimestamp startrt]=Screen('Flip', w);        
        start=GetSecs;
        while GetSecs < start+sti_Times
            tic
            cfout=fread(t_e,3, 'double');
            switch sti_stack(num_stimulus)
                case 1
                    f_eeg=cfout(1);
                case 2
                    f_eeg=cfout(1);
                case 3 % select a clf output data for foot motor imagery
                    f_eeg=cfout(FOOTfb);
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
                elseif sti_stack(num_stimulus)==3         
                    FixCross2(1,2)=FixCross2(1,2)-(f_eeg*weight(2))+bias(2);
                    FixCross2(2,2)=FixCross2(2,2)-(f_eeg*weight(2))+bias(2);
                    FixCross2(1,4)=FixCross2(1,4)-(f_eeg*weight(2))+bias(2);
                    FixCross2(2,4)=FixCross2(2,4)-(f_eeg*weight(2))+bias(2);   
                end
            end            
        end        
    end
    if ~exp_on
        break;
    end
end

ppTrigger(6)%END
Screen('TextSize',w, 24);
DrawFormattedText(w, 'Thank you.', 'center', 'center', [0 0 0]);
ShowCursor;
Screen('Flip', w);
WaitSecs(1);
Screen('CloseAll');
fclose('all');
Priority(0);


end

