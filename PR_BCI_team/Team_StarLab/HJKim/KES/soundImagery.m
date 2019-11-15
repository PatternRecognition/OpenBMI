function experiment0(rs_time, scrNum, port)
sca;
close all;

sock = tcpip('localhost', 30000, 'NetworkRole', 'Client');
set(sock, 'OutputBufferSize', 1024); % Set size of receiving buffer, if needed.

% Trying to open a connection to the server.
while(1)
    try
        fopen(sock);
        break;
    catch
        fprintf('%s \n','Cant find Server');
    end
end

global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(port);

screens = Screen('Screens');

white = WhiteIndex(scrNum);
black = BlackIndex(scrNum);
gray = (white+black)/2;

[w, windowRect] = PsychImaging('OpenWindow', scrNum, black);
[wWidth, wHeight] = Screen('WindowSize', w);

Screen('BlendFunction', w, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
Screen('TextFont', w, 'Ariel');
Screen('TextSize', w, 36);
Screen('TextStyle', w, 0);

[xCent, yCent] = RectCenter(windowRect);

% cenLineDimPix = 40;

ordInit = repmat([1, 1, 2], 1, 30); % non-target -> 1, target 2
% ordInit = ordInit(randperm(length(ordInit)));
% colOrdInit = uint8(repelem([[255 0 0]; [0 0 255]]',1,[60 30]));
colOrdInit = repmat([[255 255 255]; [255 255 255]; [255 255 255]/2], 30,1)';

centLineLength = wHeight;
centLineCoords = [0 0; -centLineLength/2 centLineLength/2];

centLineWidth = 4;
centColor = white;

xInterval = 180;
xLinesInit = kron(cumsum(repmat(xInterval, 1, 90)), ones(1,2));
yLines = repmat([-yCent yCent]/4, 1, 90);

xStep = xInterval/60;

lineWidth = 10;

%% Beep
beepLengthSecs=0.1;
rate=44100;
beepY = MakeBeep(8000,beepLengthSecs,rate);
Snd('Open');

%% fixation cross
[X,Y] = RectCenter(windowRect);
FixationSize = 40;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];

%% Resting State
Screen('TextSize',w, 50);
DrawFormattedText(w,'Closed your eyes\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,77);
Screen('Flip', w);
WaitSecs(rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'Recording Resting state\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,78);
Screen('FillRect', w, [255 255 255], FixCross');
Screen('Flip', w);
WaitSecs(rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'It will start in 3 secs','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
WaitSecs(3);
Screen('Flip', w);
HideCursor;
%%
for i = 1:10
    %     if mod(i, 2) == 1, lineColor = gray; else lineColor = black; end
    %     jit = randi([1.5, 2.5]*60);
    fprintf('End %d trial\n', i);
    ordIdx = randperm(length(ordInit));
    trigOrd = ordInit(ordIdx);
    colOrd = colOrdInit(:,ordIdx);
    
    xLines = xLinesInit;
    a = GetSecs();
    Snd('Play',beepY,rate);
    ppWrite(IO_ADD, 15);
    WaitSecs(1);

    tic;
    for j = 1:60*100
        Screen('DrawLines', w, [xLines; yLines], ...
            lineWidth, uint8(kron(colOrd, ones(1, 2))), [xCent, yCent], 2);
        %         Screen('DrawLines', w, allCoords, ...
        %             crossWidth, white, [xCent, yCent], 2);
        Screen('DrawLines', w, centLineCoords, ...
            centLineWidth, centColor, [xCent, yCent], 2);
        %         Screen('DrawingFinished', w);
        %         if any(and((xLines >= -xStep/2),(xLines < xStep/2))) -> Jittering 사라져서 주석처리
        if any(xLines == 0)
            trig = trigOrd(ceil(find(xLines==0, 1)/2));
            ppWrite(IO_ADD,trig);
            disp(toc);
            tic;
        elseif any(xLines == xStep*7)
            if trigOrd(ceil(find(xLines == xStep*7, 1)/2)) == 1
                fwrite(sock, 1);
            end
        end
        Screen('Flip', w);
        xLines = xLines - xStep;
    end
    disp(GetSecs()-a);
    fprintf('End %d trial\n', i);
    DrawFormattedText(w, 'Rest', 'center', 'center', [255 255 255]);
    Screen('Flip', w);
    GetClicks(w);
end

%% Resting State
DrawFormattedText(w,'Recording Resting state\n\nPlease follow instructions\n\nClick to start','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,78);
Screen('FillRect', w, [255 255 255], FixCross');
Screen('Flip', w);
WaitSecs(rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'THANK YOU','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(3);
ppWrite(IO_ADD, 222);
fwrite(sock, 2);
sca;
end