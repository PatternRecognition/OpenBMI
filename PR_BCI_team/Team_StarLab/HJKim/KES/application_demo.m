function [ output_args ] = application_demo( varargin )
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
    set(sock, 'OutputBufferSize', 1024);
    % Open connection to the client
    fprintf('%s \n','Client Connecting...');
    fopen(sock)
    fprintf('%s \n','Client Connected');
    set(sock,'Timeout', 0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_character = spellerText;

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

%%
% spell_char = ['A':'Z','1':'9','_'];
spell_char = {'A','B','C','D','E','F',...
                'G','H','I','J','K','L',...
                'M','N','O','P','Q','R',...
                'S','T','U','V','W','X',...
                'Y','Z',' ',' ',' ',' ',...
                ' ',' ',' ','_','SEND','ESC'};
imgs_name = cellstr(ls('C:\Users\cvpr\Desktop\Application_Demo\App\PNG\*_off.png'));
imgs = cellfun(@imread, imgs_name, 'Uni', false);
demo_images(:,1) = cellfun(@(x) Screen('MakeTexture', w, x), imgs);
imgs_name = cellstr(ls('C:\Users\cvpr\Desktop\Application_Demo\App\PNG\*_on.png'));
imgs = cellfun(@imread, imgs_name, 'Uni', false);
demo_images(:,2) = cellfun(@(x) Screen('MakeTexture', w, x), imgs);
imgs_name = cellstr(ls('C:\Users\cvpr\Desktop\Application_Demo\App\PNG\*_check.png'));
imgs = cellfun(@imread, imgs_name, 'Uni', false);
demo_images(:,3) = cellfun(@(x) Screen('MakeTexture', w, x), imgs);
tog = false(36,1);
clear imgs imgs_name
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
speller_size = [6 6];
escapeKey = KbName('esc');
waitKey=KbName('*');

highlighted_char = 0;
sock_read = 0;
chat_flag = false;
lay_char = demo_images;
layout = @demo_layout;
%% fixation cross
[X,Y] = RectCenter(rect);
FixationSize = 40;
FixCross = [X-1,Y-FixationSize,X+1,Y+FixationSize;X-FixationSize,Y-1,X+FixationSize,Y+1];
%%
rect_origin = rect;
rect = [rect(1:3) rect(3)/16*9];
rect = [rect(1) (rect_origin(4)-rect(4))/2+rect(2) rect(3) rect(4)];

text_size = ceil(rect(4)/(speller_size(1) + 2)*0.65);
loc_layout = proc_getlayout(speller_size, rect);

black = BlackIndex(w);
Screen('FillRect', w, black);
Screen('TextFont',w, 'Arial');
Screen('TextStyle', w, 0);
%% dot
dot = true;
time_delay = 0;

% HideCursor;
%% Beep
beepLengthSecs=0.1;
rate=44100;
beepY = MakeBeep(freq,beepLengthSecs,rate);
Snd('Open');
%% Start Paradgims
% Snd('Play',beepY,rate);
% pause(1);
% Snd('Play',beepY,rate);
% pause(1);
% Snd('Play',beepY,rate);
% pause(1);
% Snd('Play',beepY,rate);
% pause(1);
% Snd('Play',beepY,rate);
% pause(1);
%%
trig = 1;
% ntri = 1; %non-target trigger
% ttri = 2; %target trigger
%% Start
ppWrite(IO_ADD, 111);
n_char = 1;
while true   %%korea university 부분
    %     target_ind = find(test_character(n_char) == spell_char); %find the positions for target
    if chat_flag
        ff = highlighted_char + 100;
    else
        ff = highlighted_char;
    end
    
    if ff == 6
        lay_char = spell_char;
        layout = @spell_layout;
        chat_flag = true;
    elseif ff == 136
        lay_char = demo_images;
        layout = @demo_layout;
        chat_flag = false;
        T_char=[];  % answers
    end
    highlighted_char = 0;
    %% timer
    Screen('FillRect', offw, [0 0 0]);
    layout(offw, test_character, lay_char, loc_layout, text_size, rect, dot,highlighted_char, tog);
    
    if online && chat_flag
        if~isempty(T_char)
            Screen('TextSize',offw, ceil(text_size/3));
            Screen('DrawText', offw, T_char,0, 0, [255, 255, 255]);
        end
    end
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
    %
%     Snd('Play',beepY,rate);
%     Snd('Play',beepY,rate);
%     Snd('Play',beepY,rate);
    %
%     WaitSecs(2);
    %     target_ind=find(lay_char==test_character(n_char)); %% target highlight
    %     %     target_highlight=find(spell_char(:)==test_character(n_char)); %% target highlight
    %     Screen('TextSize',w, text_size);
    %     textbox = Screen('TextBounds', w, lay_char(target_ind));
    %     Screen('DrawText', w, lay_char(target_ind), loc_layout(target_ind,1)-(textbox(3)/2), ...
    %         loc_layout(target_ind,2)-(textbox(4)/2), [255, 255, 255]);
    %     Screen('TextSize',w, ceil(text_size/2));
    %     Screen('DrawText', w, '.', loc_layout(target_ind,1), ...
    %         loc_layout(target_ind,2)- ceil(text_size/2), [255, 255, 255]);
    %     Screen('Flip', w);
    %     Snd('Play',beepY, rate);
    %     WaitSecs(0.5);
    %     Screen('CopyWindow', offw, w);
    %     Screen('Flip', w);
    ppWrite(IO_ADD,15);  % 15 start
    WaitSecs(1);
    if online
        fread(sock,1024, 'uint8');
    end
    n_seq = 1;
    while true %nsequence 만큼 하나의 target character를 반복
        % 여기서 Highlight 받아오고
        % 새로 만들고
        %
        if n_seq > nsequence
            n_seq = 1;
        end
        Screen('TextSize',w, ceil(text_size/2));
        textbox = Screen('TextBounds', w, '.');
        for n_run=1:12       %run 6X6 speller
            Screen('CopyWindow', offw, w);
            Draw_cell = order{n_seq}(n_run,:);
            
            for j = Draw_cell %A presentation in a run
                Screen('DrawText', w, '.', loc_layout(j,1)-(textbox(3)*1.3), ...
                    loc_layout(j,2)-textbox(4)- ceil(text_size/2), [255, 255, 255]);
            end
            vbl = Screen('Flip', w, 0, 1);
            %% trigger
                        trig = sum(ismember(Draw_cell, 8))+1; %% 타겟부분일경우만 trigger
%                         if sum(trig)      %target
%                             ppWrite(IO_ADD,ttri);
%                         else            %non-target
%                             ppWrite(IO_ADD,ntri);
%                         end
            ppWrite(IO_ADD, trig);
            Screen('Flip', w, vbl + sti_Times);
            Screen('CopyWindow', offw, w);
            vbl = Screen('Flip', w, 0, 1);
            Screen('Flip', w, vbl + sti_Interval);
            if online && mod(n_run+(n_seq-1)*12,6) == 0 %&& ~all([n_seq, n_run] == [1,3])
                sock_read = fread(sock, 5,'uint8');
                if isempty(sock_read)
                    sock_read = 0;
                end
                if any(ismember(sock_read, 100))
                    break;
                end
                highlighted_char = sock_read(end);
                disp(highlighted_char);
                Screen('FillRect', offw, [0 0 0]);
                layout(offw, test_character, lay_char, loc_layout, text_size, rect, dot, highlighted_char, tog);
                if online && chat_flag
                    if~isempty(T_char)
                        Screen('TextSize',offw, ceil(text_size/3));
                        Screen('DrawText', offw, T_char,0, 0, [255, 255, 255]);
                    end
                end
            end
        end
        if any(ismember(sock_read, 100))
            fprintf('Selected index: %d\n', highlighted_char);
            break;
        end
        n_seq = n_seq + 1;
    end
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
    
    WaitSecs(1);
    ppWrite(IO_ADD,14); %online 파일의 c_"n초기화->eog switch가 안들어오도록
    
    if online
        tic;
        result = highlighted_char;
        fid = fopen('C:\Users\cvpr\Desktop\Application_Demo\Demo1\aaa.txt', 'w');%\Unity\Demo\Assets\aaa.txt', 'w');
        fwrite(fid, string(result));
        fclose(fid);
        if chat_flag
            if any(result == [0 , 100])
                result = 1;
            end
            if any(result == [35, 36])
                T_char = [];
            else
                str = spell_char{result};
                if isequal(str,'_')
                    str = ' ';
                end
                T_char=[T_char str];
                clear str
            end
        else
            if ~any(result == [6, 31:36])
                tog(result) = ~tog(result);
            end
        end
        fread(sock,1024);
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
    
    %     if n_char == ceil(length(test_character)/2 )
    %         DrawFormattedText(w, 'Left Mouse click three times to restart an experiment', 'center', 'center', [255 255 255]);
    %         Screen('Flip', w);
    %         GetClicks(w);
    %         GetClicks(w);
    %         GetClicks(w);
    %     end
    n_char = n_char + 1;
end
% %% Resting State
% Screen('TextSize',w, 50);
% DrawFormattedText(w,'Recording Resting state\n\nPlease follow instructions','center','center',[255 255 255]);
% Screen('Flip', w);
% GetClicks(w);
% ppWrite(IO_ADD,78);
% Screen('FillRect', w, [255 255 255], FixCross');
% Screen('Flip', w);
% WaitSecs(rs_time);
% ppWrite(IO_ADD, 14);
% DrawFormattedText(w,'Thank you','center','center',[255 255 255]);
% Screen('Flip', w);
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