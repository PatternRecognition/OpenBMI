function [ weight, bias ] = MI_setting(type)
% server eeg
t_e = tcpip('localhost', 3000, 'NetworkRole', 'Server');
set(t_e, 'InputBufferSize', 3000);
% Open connection to the client.
fopen(t_e);
fprintf('%s \n','Client Connected');
connectionServer = t_e;
set(connectionServer,'Timeout',.1);

escapeKey = KbName('esc');
waitKey=KbName('s');
resetKey=KbName('r');
weight_up=KbName('up');
weight_down=KbName('down');
bias_plus=KbName('right');
bias_minus=KbName('left');

Screen('Preference', 'SkipSyncTests', 1);
% screen setting (gray)
screenRes = [0 0 640 480];
screens=Screen('Screens');
screenNumber=max(screens);
gray=GrayIndex(screenNumber);
[w, wRect]=Screen('OpenWindow',screenNumber, gray);
% ScreenP = Psychtoolbox_Open_Kb(screenNumber,fixSize);
[X,Y] = RectCenter(wRect);
FixationSize = 20;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];
Screen('FillRect', w, [255 0 0], FixCross');
Screen('Flip', w);
cal_on=true;
weight=0.2;
bias=0;
while cal_on
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            cal_on=false;
        elseif keyCode(waitKey)
            warning('stop')
            GetClicks(w);               
        elseif keyCode(weight_up)
            weight=weight+0.1;
        elseif keyCode(weight_down)
            if weight<0
                weight=0;
            else
                weight=weight-0.1;
            end            
        elseif keyCode(bias_plus)
            bias=bias+0.1;
        elseif keyCode(bias_minus)
            bias=bias-0.1;
        elseif keyCode(resetKey)
            FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];
        else 
        end
    end
    
    f_eeg=fread(t_e,1,'double')
    if ~isempty(f_eeg)
        str=sprintf('weight=%f, bias=%f',weight, bias);
        Screen('TextSize',w,50); % smkim
        DrawFormattedText(w, str, 0, 0, [0, 0, 0, 255]);
        Screen('FillRect', w, [255 0 0], FixCross');
        Screen('Flip', w);
        switch type % smkim
            case 1
                FixCross(1,1)=FixCross(1,1)-(f_eeg*weight)+bias;
                FixCross(2,1)=FixCross(2,1)-(f_eeg*weight)+bias;
                FixCross(1,3)=FixCross(1,3)-(f_eeg*weight)+bias;
                FixCross(2,3)=FixCross(2,3)-(f_eeg*weight)+bias;
            case 2  % smkim
                FixCross(1,2)=FixCross(1,2)+(f_eeg*weight)-bias;
                FixCross(2,2)=FixCross(2,2)+(f_eeg*weight)-bias;
                FixCross(1,4)=FixCross(1,4)+(f_eeg*weight)-bias;
                FixCross(2,4)=FixCross(2,4)+(f_eeg*weight)-bias;
        end
    end
end

Screen('CloseAll');

end

