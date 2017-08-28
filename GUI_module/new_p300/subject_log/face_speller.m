function [ output_args ] = Hybrid_P300_online4( varargin )
%HYBRID_P300 Summary of this function goes here
%   Detailed explanation goes here
%% online
% % trigger
global IO_LIB;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('D010');
%%%
%%
if isempty(varargin) || varargin{1}==-1 %training
    varargin{1}=-1;
    run_eog=false;
    run_feedback=false;
else
    if varargin{1}==1 %eog
        run_eog=true;
        run_feedback=true;
    else   % test
        run_eog=false;
        run_feedback=true;
    end
    
    if length(varargin)>1
        FILE=varargin{2};
        switch varargin{1}
            case -1
                FILE_=[varargin{2} varargin{3}];
            case 0
                FILE_=[varargin{2} varargin{3}];
            case 1
                FILE_=[varargin{2} varargin{3}];
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
waitKey=KbName('space');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(varargin) || varargin{1}==-1  %training procedure
    test_character_show = ['BRAIN_COMPUTER_INTERFACE'];
    test_character = ['BRAIN_COMPUTER_INTERFACE'];
else  %test procedures
    test_character_show = ['KOREA'];
    test_character = ['KOREA'];
end

% load cell_order;
% cell_order_all=cell_order_all(1:length(test_character),:,:,:); % 글자 수만 가져오기
load cell_order_new;
nsequence = 10; %각 글자당 sequence 수
T_char=[];  % answers
eog_target=[];
eog_best=[];
%%

screenNum = 2; %For Laptop
[w, rect] = Screen('OpenWindow', screenNum,[], [0 0 400 600]);
% [w, rect] = Screen('OpenWindow', screenNum);
loc_layout = proc_getlayout(speller_size, rect);
black = BlackIndex(w);
white = WhiteIndex(w);
Screen('FillRect', w, black);

% face
subject_img = ['face_2.png'];
self_img = imread(subject_img);
textureIndex = Screen('MakeTexture', w, self_img);

%%Time
t_text_size = 60;
n_text_size = 50;
face_size = t_text_size+70;

sti_Times = 0.135;%per/S 10 %%여기 다시한번 체크할것
sti_Interval = 0.05;%per/S

vbl = Screen(w, 'Flip');
ifi = Screen('GetFlipInterval', w);

% slack=ifi/2; %% 시작부분 timer
count_speed = 1.8; %count-down speed
normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
% load cell_order

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
    
    %     ppWrite(IO_ADD,15);  % 15 start
    tic
    for n_seq = 1:nsequence %nsequence 만큼 하나의 target character를 반복
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                Screen('CloseAll');
                fclose('all');
            elseif keyCode(waitKey)
                GetClicks(w);
            end
        end
        
        
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
                        %                         Screen('TextSize',w, t_text_size);
                        %                         textbox = Screen('TextBounds', w, spell_char(j));
                        %                         Screen('DrawText', w, spell_char(j), loc_layout(j,1)-(textbox(3)/2), ...
                        %                             loc_layout(j,2)-(textbox(4)/2), [255, 255, 255]);
                        Screen('DrawTextures', w, textureIndex, [], [loc_layout(j,1)-(face_size/2)+10 ...
                            loc_layout(j,2)-(face_size/2) ...
                            loc_layout(j,1)-(face_size/2)+face_size-10 ...
                            loc_layout(j,2)-(face_size/2)+face_size], 0, 0);
                    else %non-target character
                        Screen('TextSize',w, n_text_size);
                        textbox = Screen('TextBounds', w, spell_char(j));
                        Screen('DrawText', w, spell_char(j), loc_layout(j,1)-(textbox(3)/2), ...
                            loc_layout(j,2)-(textbox(4)/2), [100, 100, 100]);
                    end
                end
            end
            if run_eog  %깜빡 거리는거 제거
                if ~isempty(eog_target)
                    if eog_target(1) ~= 0
                                            Screen('TextSize',w, n_text_size);
                            textbox = Screen('TextBounds', w, spell_char(eog_target(1)));
                            Screen('DrawText', w, spell_char(eog_target(1)), loc_layout(eog_target(1),1)-(textbox(3)/2), ...
                                loc_layout(eog_target(1),2)-(textbox(4)/2), [255, 255, 255]);
%                         for n=1:length(eog_target)
%                             target_index=eog_target(n);
%                             Screen('TextSize',w, 35);
%                             textbox = Screen('TextBounds', w,  int2str(n));
%                             Screen('DrawText', w, int2str(n), loc_layout(target_index,1)-(textbox(3)/2), ...
%                                 loc_layout(target_index,2)-(textbox(4)/2)-60, [255, 255, 255]);
%                         end
                    end
                end
            end
            %% trigger
            if varargin{1}==-1 || isempty(varargin)
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
            %             %% information
            %             Screen('TextSize',w, ceil(n_text_size/2));
            %             infor=['trial:' int2str(n_char),'  ' ,'sequence:' int2str(n_seq),'  ', 'run:' int2str(n_run)];
            %             Screen('DrawText', w, infor, 1200, 45, [255, 255, 255]);
            %% feedback

            if run_eog
                if n_seq ~= 1
                    fileID = fopen(['subject_log\' FILE '_prior.txt'], 'r'); %
                    formatSpec = '%f';
                    A = fscanf(fileID,formatSpec);
                    eog_target=A';
                    
                    fclose(fileID);
                    if ~isempty(eog_target)
                        eog_best=spell_char2{eog_target(1)};
                        if eog_target(1) ~= 0
                            Screen('TextSize',w, n_text_size);
                            textbox = Screen('TextBounds', w, spell_char(eog_target(1)));
                            Screen('DrawText', w, spell_char(eog_target(1)), loc_layout(eog_target(1),1)-(textbox(3)/2), ...
                                loc_layout(eog_target(1),2)-(textbox(4)/2), [255, 255, 255]);
%                             for n=1:length(eog_target)
%                                 target_index=eog_target(n);
%                                 Screen('TextSize',w, 35);
%                                 textbox = Screen('TextBounds', w,  int2str(n));
%                                 Screen('DrawText', w, int2str(n), loc_layout(target_index,1)-(textbox(3)/2), ...
%                                     loc_layout(target_index,2)-(textbox(4)/2)-60, [255, 255, 255]);
%                             end
                          
                        end
                    end
                end
            end
            %%
            if run_eog   %eog로 선택하는 부분
                fileID2 = fopen(['subject_log\' FILE '_switch.txt'], 'r');
                formatSpec = '%f';
                A2 = fscanf(fileID2,formatSpec);
                fclose(fileID2);
                
                fileID3 = fopen(['subject_log\' FILE '_switch.txt'], 'w'); % matlab 보내기
                fprintf(fileID3,'%5d\n',0);
                fclose(fileID3);
                if A2==1;
                    break
                end
            end
            
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
            
            if run_eog  %깜빡 거리는거 제거S
                if ~isempty(eog_target)
                    if eog_target(1) ~= 0
                                                 Screen('TextSize',w, n_text_size);
                            textbox = Screen('TextBounds', w, spell_char(eog_target(1)));
                            Screen('DrawText', w, spell_char(eog_target(1)), loc_layout(eog_target(1),1)-(textbox(3)/2), ...
                                loc_layout(eog_target(1),2)-(textbox(4)/2), [255, 255, 255]);
%                         for n=1:length(eog_target)
%                             target_index=eog_target(n);
%                             Screen('TextSize',w, 35);
%                             textbox = Screen('TextBounds', w,  int2str(n));
%                             Screen('DrawText', w, int2str(n), loc_layout(target_index,1)-(textbox(3)/2), ...
%                                 loc_layout(target_index,2)-(textbox(4)/2)-60, [255, 255, 255]);
%                         end
                    end
                end
            end
            
            start_3 = Screen('Flip', w, start_2 + sti_Interval);
            start_1 = start_3;    
        end
        
        if run_eog   %eog로 선택하는 부분
            if A2
                break;
            end
        end
        %         if run_feedback
        %             if n_seq==10 && n_run==12
        %                 fileID2 = fopen(['subject_log\' FILE '_char.txt'], 'r'); %
        %                 formatSpec2 = '%c';
        %                 str = fscanf(fileID2,formatSpec2);
        %                 fclose(fileID2);
        %                 SAVE{n_char}.char=str;
        %                 SAVE{n_char}.time=toc;
        %             end
        %         end
      
    end
    
    
    if run_eog   %eog로 선택하는 부분
        if A2
            A2=0;
        end
    end
    
    
    normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
    if run_feedback
        if run_eog
            str = eog_best %fscanf(fileID2,formatSpec2);
            T_char=[T_char str];
            SAVE{n_char}.char=str;
            SAVE{n_char}.time=toc; %   WaitSecs(2); 있으므로 -2s 해주기 희진-영태-상준 +2초씩 해줄것
        end
    end
    WaitSecs(2);
    
    if run_feedback
        if ~run_eog
            fileID2 = fopen(['subject_log\' FILE '_char.txt'], 'r'); %
            formatSpec2 = '%c';
            str=fscanf(fileID2,formatSpec2);
            fclose(fileID2);
            T_char=[T_char str];
            SAVE{n_char}.char=str;
            SAVE{n_char}.time=toc-2; %   WaitSecs(2); 있으므로 -2s 해주기
        end
    end
    
    Screen('Flip', w);
    ppWrite(IO_ADD,14); %online 파일의 c_"n초기화->eog switch가 안들어오도록
    
end
ppWrite(IO_ADD,20); %15=end
if run_feedback
    save(['subject_log\' FILE_ '_MATLAB1_DATA'], 'SAVE');
end
pause(1);
ppWrite(IO_ADD,20); %15=end
% ppWrite(IO_ADD,16); %finish
Screen('CloseAll');
fclose('all');
end

