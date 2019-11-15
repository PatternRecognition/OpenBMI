function [ output_args ] = application_train( varargin )
% Face_speller Summary of this function goes here
%   Detailed explanation goes here
% txt = 'NEURAL_NETWORKS_AND_DEEP_LEARNING'; n_seq = 5; screens = Screen('Screens'); n_scr = max(screens); stimulus_time=0.135; interval_time=0.05; small = true;
% small_dot_speller({'exp_type',1; 'port', '4FF8';'text',txt; 'nSequence',n_seq;'screenNum',n_scr;'sti_Times',stimulus_time;'sti_Interval',interval_time; 'small', small});
%
%% Init
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'port'), error('No input port information');
else port= opt.port;end

if ~isfield(opt,'screenSize'),screenSize='full';
elseif ischar(opt.screenSize), screenSize = opt.screenSize;
else screenSize=[0,0,opt.screenSize];end
if ~isfield(opt,'text'),spellerText='DEFAULT_TEXT';
else spellerText=upper(opt.text);end

if ~isfield(opt,'nSequence'), nSequence=10;
else nSequence= opt.nSequence;end
if ~isfield(opt,'screenNum'), screenNum=2;
else screenNum= opt.screenNum;end

if ~isfield(opt,'sti_Times'),sti_Times=0.135;
else sti_Times= opt.sti_Times;end

if ~isfield(opt,'sti_Interval'),sti_Interval=0.05;
else sti_Interval= opt.sti_Interval;end

if ~isfield(opt, 'resting'), rs_time = 60;
else rs_time = opt.resting; end

if ~isfield(opt, 'frequency'), freq = 0;
else freq = opt.frequency; end

if ~isfield(opt, 'online'), online = false;
else online = opt.online; end

% % trigger
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(opt.port);

%% check for connection
if online
    sock = tcpip('localhost', 30000, 'NetworkRole', 'Server');
    set(sock, 'InputBufferSize', 1024);
    % Open connection to the client
    fprintf('%s \n','Client Connecting...');
    fopen(sock)
    fprintf('%s \n','Client Connected');
    set(sock,'Timeout',1.5);
end

spell_char = ['A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
speller_size = [6 6];
escapeKey = KbName('esc');
waitKey=KbName('*');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_character = spellerText;

% load cell_order;
% cell_order_all=cell_order_all(1:length(test_character),:,:,:); % 글자 수만 가져오기
order = importdata('C:\Users\cvpr\Desktop\Application_Demo\App\random_order_v3.mat');
nsequence = nSequence; %각 글자당 sequence 수
T_char=[];  % answers
eog_target=[];
eog_best=[];
%%
if ischar(screenSize) && strcmp(screenSize,'full')
    [w, rect] = Screen('OpenWindow', screenNum);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
else
    [w, rect] = Screen('OpenWindow', screenNum,[], screenSize);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
end

%% fixation cross
[X,Y] = RectCenter(rect);
FixationSize = 40;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];
%%
rect_origin = rect;
rect = [rect(1:3) rect(3)/16*9];
rect = [rect(1) (rect_origin(4)-rect(4))/2+rect(2) rect(3) rect(4)];

%%text_size
text_size = ceil(rect(4)/(speller_size(1) + 2)*0.65);
loc_layout = proc_getlayout(speller_size, rect);

black = BlackIndex(w);
Screen('FillRect', w, black);    
Screen('TextFont',w, 'Arial');
Screen('TextStyle', w, 0);

%% dot
dot = true;
time_delay = 0;

HideCursor;
%% Beep
beepLengthSecs=0.1;
rate=44100;
beepY = MakeBeep(freq,beepLengthSecs,rate);
Snd('Open');
%% Start Paradgims
Snd('Play',beepY,rate);
pause(1);
Snd('Play',beepY,rate);
pause(1);
Snd('Play',beepY,rate);
pause(1);
Snd('Play',beepY,rate);
pause(1);
Snd('Play',beepY,rate);
pause(1);
%% Start
ppWrite(IO_ADD, 111);
for n_char = 1:length(test_character)   %%korea university 부분
%     target_ind = find(test_character(n_char) == spell_char); %find the positions for target
    %% timer
    Screen('FillRect', offw, [0 0 0]);
    normal_layout(offw, test_character, spell_char, loc_layout, text_size, rect, dot);
    Screen('TextSize',offw, ceil(text_size/1.5));
    Screen('DrawText', offw, test_character(n_char), (rect(3)/2), ceil(text_size/3), [255, 255, 255]);
    if online
        if~isempty(T_char)
            Screen('TextSize',offw, ceil(text_size/3));
            Screen('DrawText', offw, T_char,0, ceil(text_size/3*2), [255, 255, 255]);
        end
    end
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
    
    Snd('Play',beepY, rate);
    
    WaitSecs(2);    
   target_ind=find(spell_char==test_character(n_char)); %% target highlight
%     target_highlight=find(spell_char(:)==test_character(n_char)); %% target highlight
    Screen('TextSize',w, text_size);
    textbox = Screen('TextBounds', w, spell_char(target_ind));
    Screen('DrawText', w, spell_char(target_ind), loc_layout(target_ind,1)-(textbox(3)/2), ...
        loc_layout(target_ind,2)-(textbox(4)/2), [255, 255, 255]);
    Screen('TextSize',w, ceil(text_size/2));
    Screen('DrawText', w, '.', loc_layout(target_ind,1), ...
                    loc_layout(target_ind,2)- ceil(text_size/2), [255, 255, 255]);
    Screen('Flip', w);
    Snd('Play',beepY, rate);
    WaitSecs(0.5);
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
    ppWrite(IO_ADD,15);  % 15 start
    WaitSecs(2);
    
    for n_seq = 1:nsequence %nsequence 만큼 하나의 target character를 반복
        for n_run=1:12       %run 6X6 speller
            Screen('CopyWindow', offw, w);
            Draw_cell = order{n_seq}(n_run,:);
            
            for j = Draw_cell %A presentation in a run -->>여기도 점위치 변경
                Screen('DrawText', w, '.', loc_layout(j,1), loc_layout(j,2) - 1.2 * ceil(text_size), [255, 255, 255]);
            end
            vbl = Screen('Flip', w, 0, 1);
            %% trigger
            trig = ismember(Draw_cell, target_ind); %% 타겟부분일경우만 trigger
            if sum(trig)      %target
                ppWrite(IO_ADD,2);
            else            %non-target
                ppWrite(IO_ADD,1);
            end
            Screen('Flip', w, vbl + sti_Times);
            Screen('CopyWindow', offw, w);
            vbl = Screen('Flip', w, 0, 1);
            Screen('Flip', w, vbl + sti_Interval);
        end
    end    
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
    
    WaitSecs(1);
    ppWrite(IO_ADD,14); %online 파일의 c_"n초기화->eog switch가 안들어오도록
    
     if online
        tic;
        str = '*';
        result_str = fread(sock, 1);
        if ~isempty(result_str)
            str = char(result_str);
        end
        T_char=[T_char str];
        clear str
        time_delay = toc;
    end
    
    tic;
    while toc < 2 - time_delay
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(1);
                Screen('CloseAll');
                ppWrite(IO_ADD,20); %15=end
                fclose('all');
                output_args = 'Re execute paradigm (Already connected with client)...';
                return;
            elseif keyCode(waitKey)
                DrawFormattedText(w, 'Left Mouse click three times to restart an experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                GetClicks(w);
                GetClicks(w);
                GetClicks(w);
            end
        end
    end
    
    if n_char == ceil(length(test_character)/2 )
        DrawFormattedText(w, 'Left Mouse click three times to restart an experiment', 'center', 'center', [255 255 255]);
        Screen('Flip', w);
        GetClicks(w);
        GetClicks(w);
        GetClicks(w);
    end
    
end
%% Resting State
Screen('TextSize',w, 50);
DrawFormattedText(w,'Recording Resting state\n\nPlease follow instructions','center','center',[255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppWrite(IO_ADD,78);
Screen('FillRect', w, [255 255 255], FixCross');
Screen('Flip', w);
WaitSecs(rs_time);
ppWrite(IO_ADD, 14);
DrawFormattedText(w,'Thank you','center','center',[255 255 255]);
Screen('Flip', w);
%%

ppWrite(IO_ADD,222); %15=end
GetClicks(w);
pause(1);
% ppWrite(IO_ADD,20); %15=end
% ppWrite(IO_ADD,16); %finish
sca;
fclose('all');

output_args = 'Done all process...';

end