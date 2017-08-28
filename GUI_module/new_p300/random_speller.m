function [ output_args ] = random_speller( varargin )
% Random_speller Summary of this function goes here
%   Detailed explanation goes here
%% Init
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'port'), error('No input port information');
else port= opt.port;end

if ~isfield(opt,'screenSize'),screenSize='full';
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
%% check online connection
global sock
    if varargin{1,1}{1,2}==1 % online
         flushinput(sock);
      % check for connection
        a = fread(sock,1);       
        if isempty(a)
            output_args = 'Make sure to make the online connection';
            return;
        end
    elseif isempty(varargin) || varargin{1,1}{1,2}==-1 %offline
                
    end
%% online
% % trigger
global IO_LIB;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(opt.port);
%%%
global direct
direct=fileparts(which('OpenBMI'));
%%
if isempty(varargin) || varargin{1,1}{1,2}==-1 %training
    varargin{1,1}{1,2}=-1;
    run_feedback=false;
else
    if varargin{1,1}{1,2}==1 
        run_feedback=true;
    else   % test
        run_feedback=true;
    end
    
    if length(varargin)>1
        FILE=varargin{1,1}{2,2};
        switch varargin{1,1}{1,2}
            case -1
                FILE_=[varargin{1,1}{2,2} varargin{1,1}{3,2}];
            case 0
                FILE_=[varargin{1,1}{2,2} varargin{1,1}{3,2}];
            case 1
                FILE_=[varargin{1,1}{2,2} varargin{1,1}{3,2}];
        end
    end
end

spell_char = ['A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'];

spell_char2 = {'A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'};

spell_num=[1, 2, 3, 4, 5, 6; 7,8,9,10,11,12;13,14,15,16,17,18;19,20,21,22,23,24;25,26,27,28,29,30;31,32,33,34,35,36; ...
    1,7,13,19,25,31;2,8,14,20,26,32;3,9,15,21,27,33;4,10,16,22,28,34;5,11,17,23,29,35;6,12,18,24,30,36]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
speller_size = [6 6];
escapeKey = KbName('esc');
waitKey=KbName('*');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(varargin) || varargin{1,1}{1,2}==-1  %training procedure
        test_character_show = spellerText;
        test_character = spellerText;
else  %test procedures
    test_character_show = spellerText;
    test_character = spellerText;
end

% load cell_order;
load random_cell_order;
nsequence = nSequence; %각 글자당 sequence 수
T_char=[];  % answers
eog_target=[];
eog_best=[];
%%

if ischar(screenSize) && strcmp(screenSize,'full')
    Screen('Preference', 'SkipSyncTests', 1);       % 주의
    [w, rect] = Screen('OpenWindow', screenNum );
else
    Screen('Preference', 'SkipSyncTests', 1);       % 주의
    [w, rect] = Screen('OpenWindow', screenNum,[], screenSize);
    % [w, rect] = Screen('OpenWindow', screenNum,[], [0 0 1280 720]);
end

loc_layout = proc_getlayout(speller_size, rect);
black = BlackIndex(w);
white = WhiteIndex(w);
Screen('FillRect', w, black);


%%Time
t_text_size = 60;
n_text_size = 50;

sti_Times = sti_Times;%per/S 10 
sti_Interval = sti_Interval;%per/S 

vbl = Screen(w, 'Flip');
ifi = Screen('GetFlipInterval', w);

count_speed = 1.8; %count-down speed
normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);

for n_char = 1:length(test_character)   %%korea university 부분
    target_ind = find(test_character(n_char) == spell_char); %find the positions for target
    %% timer
    normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, ceil(t_text_size));
    Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
    
    if run_feedback
        if ~isempty(T_char)
            Screen('TextFont',w, 'Arial');
            Screen('TextSize',w, ceil(n_text_size/2.5));
            Screen('TextStyle', w, 0);
            Screen('DrawText', w, regexprep(T_char,'\W',''), 10, 30, [255, 255, 255]);
        end
    end
    Screen('Flip', w);
    pause(2);
    
    i2=strcmpi(spell_char2,test_character(n_char)); %% target highlight
    Screen('TextSize',w, n_text_size);
    textbox = Screen('TextBounds', w, spell_char(i2));
    Screen('DrawText', w, spell_char(i2), loc_layout(i2,1)-(textbox(3)/2), ...
        loc_layout(i2,2)-(textbox(4)/2), [200, 200, 200]);
    Screen('Flip', w);
    pause(0.5);
    
    normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, ceil(t_text_size));
    Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
    
    if run_feedback
        if ~isempty(T_char)
            Screen('TextFont',w, 'Arial');
            Screen('TextSize',w, ceil(n_text_size/2.5));
            Screen('TextStyle', w, 0);
            Screen('DrawText', w, regexprep(T_char,'\W',''), 10, 30, [255, 255, 255]);
        end
    end
    start_1 = Screen('Flip', w, 0 + count_speed);
    pause(2);
    
        ppWrite(IO_ADD,15);  % 15 start
    tic
    for n_seq = 1:nsequence %nsequence 만큼 하나의 target character를 반복
     
        
        for n_run=1:12       %run 6X6 speller
            
            Screen('TextFont',w, 'Arial');
            Screen('TextSize',w, ceil(n_text_size/2.5));
            Screen('TextStyle', w, 0);
            Screen('DrawText', w, test_character_show, 10, 10, [255, 255, 255]);
            Screen('DrawLine', w, [50 50 50], 0, 140, 1920, 140, 5);
            Screen('TextFont',w, 'Arial');
            Screen('TextStyle', w, 0);
            Screen('TextSize',w, ceil(t_text_size));
            Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
            
            Draw_cell = cell_order{n_seq}(n_run,:);
            
            for j = 1:length(spell_char2) %A presentation in a run
                Screen('TextFont',w, 'Arial');
                Screen('TextStyle', w, 0);
                if j ~= length(spell_char)+1
                    if ~isempty(find(Draw_cell == j)) %target character
                        %% revised oyeon
                       Screen('TextSize',w, t_text_size);
                        textbox = Screen('TextBounds', w, spell_char(j));
                        Screen('DrawText', w, spell_char(j), loc_layout(j,1)-(textbox(3)/2), ...
                            loc_layout(j,2)-(textbox(4)/2), [255, 255, 255]);
                        
                    else %non-target character
                        Screen('TextSize',w, n_text_size);
                        textbox = Screen('TextBounds', w, spell_char(j));
                        Screen('DrawText', w, spell_char(j), loc_layout(j,1)-(textbox(3)/2), ...
                            loc_layout(j,2)-(textbox(4)/2), [100, 100, 100]);
                    end
                end
            end

            %% trigger
            if varargin{1,1}{1,2}==-1 || isempty(varargin)
                trig = ismember(Draw_cell, target_ind); %% 타겟부분일경우만 trigger
                if sum(trig)      %target
                    ppWrite(IO_ADD,1);
                else            %non-target
                    ppWrite(IO_ADD,2);
                end
            else
                %                 t_num=find(sum(ismember( spell_num,Draw_cell)')==6);
                ppWrite(IO_ADD,n_run);
            end
            
            %% online
            if run_feedback
                if ~isempty(T_char)
                    Screen('TextFont',w, 'Arial');
                    Screen('TextSize',w, ceil(n_text_size/2.5));
                    Screen('TextStyle', w, 0);
                    Screen('DrawText', w, regexprep(T_char,'\W',''), 10, 30, [255, 255, 255]);
                end
            end
                                    %% information
%                         Screen('TextSize',w, ceil(n_text_size/2));
%                         infor=['trial:' int2str(n_char),'  ' ,'sequence:' int2str(n_seq),'  ', 'run:' int2str(n_run)];
%                         Screen('DrawText', w, infor, 1200, 45, [255, 255, 255]);
            %% feedback           
            start_2 = Screen('Flip', w, start_1 + sti_Times);
            normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
            Screen('TextFont',w, 'Arial');
            Screen('TextStyle', w, 0);
            Screen('TextSize',w, ceil(t_text_size));
            Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
            
            if run_feedback
                if ~isempty(T_char)
                    Screen('TextFont',w, 'Arial');
                    Screen('TextSize',w, ceil(n_text_size/2.5));
                    Screen('TextStyle', w, 0);
                    Screen('DrawText', w, regexprep(T_char,'\W',''), 10, 30, [255, 255, 255]);
                end
            end
                        
            start_3 = Screen('Flip', w, start_2 + sti_Interval);
            start_1 = start_3;    
        end             
    end

    normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
%     WaitSecs(2);
    
    if run_feedback

            fileID2 = fopen([direct '\log\_char.txt'],'r'); % selected  char matlab 보내기
%             fileID2 = fopen([pwd '\OpenBMI-master\log\_char.txt'], 'r'); %
            formatSpec2 = '%c';
            str=fscanf(fileID2,formatSpec2);
            fclose(fileID2);
            T_char=[T_char str];
            SAVE{n_char}.char=str;
            SAVE{n_char}.time=toc-2; %   WaitSecs(2); 있으므로 -2s 해주기
    end
    
    Screen('Flip', w);
    ppWrite(IO_ADD,14); %online 파일의 c_"n초기화->eog switch가 안들어오도록
 %%
tic
while toc < 2
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
            Screen('Flip', w);
            WaitSecs(1);
            Screen('CloseAll');
            ppTrigger(20);
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

end
ppWrite(IO_ADD,20); %15=end
if run_feedback
%     save([pwd '\OpenBMI-master\log\SAVE'], 'SAVE');   
save(fullfile(direct, 'log',sprintf('%d%02d%02d_%02d.%02d_p300_param.mat',c(1:5))),'SAVE');    
end
pause(1);
ppWrite(IO_ADD,20); %15=end
Screen('CloseAll');
fclose('all');

        output_args = 'Done all process...';
end

