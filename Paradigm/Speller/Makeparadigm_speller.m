function [flickering_order]=Makeparadigm_speller( opt,varargin )
% Makeparadigm_speller (Experimental paradigm):
% 
% Description:
%   Basic p300 speller experiment paradigm using psychtoolbox.
%   It shows 6x6 speller.
% 
% Example:
%   Makeparadigm_speller({'text','MACHINE_LEARNING'},-1)
% 
% Input: (Nx2 size, cell-type)
%   text - text you want to write
% 
global IO_LIB IO_ADD
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('D010');

opt=opt_cellToStruct(opt);

%%

spell_char = ['A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'];

speller_size = [6 6];
t_text_size = 60;
n_text_size = 50;
escapeKey = KbName('esc');
waitKey=KbName('s');

test_character_show = opt.text;
test_character = opt.text;

%% Random flicker order
spell_num=reshape(1:36,[6,6]);
flickering_order=zeros(12,10,length(test_character)); % (#flick/seq)x(#seq)x(#char)
cell_order_all=zeros(length(flickering_order(:)),6);
for ch=1:length(test_character)
    flickering_order(:,:,ch)=Shuffle(repmat([1:12]',[1,10])); % 각 글자마다 120번씩 flickering
end
for fk=1:length(flickering_order(:))
    if flickering_order(fk)<=6      % 1~6까진 행
        cell_order_all(fk,:)= spell_num(flickering_order(fk),:);
    else                            % 7~12는 열
        cell_order_all(fk,:)= spell_num(:,(flickering_order(fk)-6));
    end
end




screenNum =2;
[w, rect] = Screen('OpenWindow', screenNum);
% [w, rect] = Screen('OpenWindow', screenNum,[], [0 0 640 480]);
% [w, rect] = Screen('OpenWindow', screenNum,[0 0 640 480]);
% [w, rect] = Screen('OpenWindow', screenNum,[0 0 1680 1050]);
loc_layout = proc_getlayout(speller_size, rect);
black = BlackIndex(w);
Screen('FillRect', w, black);

%%Time
Screen(w, 'Flip');
ifi = Screen('GetFlipInterval', w);


%%

p300_interval=0;
p300_sti=0;


target_on=false; % p300 target
count_sequence=0; % p300 한 시퀀스 12
n_sequence=0;
count_run=0; % p300 한줄 6
% cell_order_all=[];

run_p300=true;

slack=ifi/2; %% 시작부분 timer
begin_Times = 2; % 시작 전 시간
count_speed = 1; % count-down speed
normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);

begin_onset = Screen('Flip',w);
% load cell_order

pause(1);
ppWrite(IO_ADD,111);

for cur_char = 1:length(test_character)
    target_ind = find(test_character(cur_char) == spell_char); %find the positions for target
    %% timer
    normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, ceil(t_text_size));
    Screen('DrawText', w, test_character(cur_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
    start_0 = Screen('Flip', w, begin_onset + begin_Times - slack);
    
    
    %% target글자 보여주기
    normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size); %smkim(추가)
    Screen('TextFont',w, 'Arial');
    Screen('TextSize',w, ceil(n_text_size/2.5));
    Screen('TextStyle', w, 0);
    Screen('DrawText', w, test_character_show, 10, 10, [255, 100, 100]);
    Screen('DrawLine', w, [50 50 50], 0, 140, 1920, 140, 5);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, ceil(t_text_size));
    Screen('DrawText', w, test_character(cur_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, n_text_size);
%     target_char=find(ismember(spell_char,cur_char));
    target_box = Screen('TextBounds', w, spell_char(target_ind));
    Screen('DrawText', w, spell_char(target_ind), loc_layout(target_ind,1)-(target_box(3)/2), ...
        loc_layout(target_ind,2)-(target_box(4)/2), [255, 255, 255]);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, n_text_size);
    Screen('Flip',w);
    WaitSecs(2)
    
    
    
    
    for i = 3:-1:1
        normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size);
        Screen('TextFont',w, 'Arial');
        Screen('TextStyle', w, 0);
        Screen('TextSize',w, ceil(t_text_size));
        Screen('DrawText', w, test_character(cur_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
        Screen('TextFont',w, 'Arial'); % Countdown
        Screen('TextStyle', w, 0);
        Screen('TextSize',w, ceil(t_text_size)/2);
        Screen('DrawText', w, int2str(i), rect(3)-100-(ceil(t_text_size)/2), 30, [0, 0, 255]);
        start_1 = Screen('Flip', w, start_0 + count_speed - slack);
        start_0 = start_1;
    end
    %%
    prevVbl = start_0;
    run=true;
    tic
    while run,
        
        %% key check
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
%                 run=false;
                Screen('CloseAll');
                fclose('all');
                return
            elseif keyCode(waitKey)
                GetClicks(w);
            end
        end
        
        normal_layout(w, test_character_show, spell_char, loc_layout, n_text_size); %smkim(추가)
        
        Screen('TextFont',w, 'Arial');
        Screen('TextSize',w, ceil(n_text_size/2.5));
        Screen('TextStyle', w, 0);
        Screen('DrawText', w, test_character_show, 10, 10, [255, 100, 100]);
        Screen('DrawLine', w, [50 50 50], 0, 140, 1920, 140, 5);
        Screen('TextFont',w, 'Arial');
        Screen('TextStyle', w, 0);
        Screen('TextSize',w, ceil(t_text_size));
        Screen('DrawText', w, test_character(cur_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
        
        %%
        if run_p300
            Screen('TextFont',w, 'Arial');
            Screen('TextStyle', w, 0);
            Screen('TextSize',w, n_text_size);
            
            if(p300_interval >=0.2)   % p300 isi
                p300_interval=0;
                count_run=count_run+1;  % run count
                count_sequence=count_sequence+1;
                
                target_on=true;
                a = ismember(cell_order_all(count_run,:), target_ind); %% 타겟부분일경우만 trigger
                if sum(a)       % target
                    ppWrite(IO_ADD,1);
                else            % non-target
                    ppWrite(IO_ADD,2);
                end
            end
            
            if target_on
                p300_sti=p300_sti+sti; sti=0;
                if p300_sti<0.09 % 0.09동안만 타겟이 들어오도록  **   p300 sti
                    for j = 1:6 % target
                        str_num=cell_order_all(count_run,j);
                        textbox = Screen('TextBounds', w, spell_char(str_num));
                        Screen('DrawText', w, spell_char(str_num), loc_layout(str_num,1)-(textbox(3)/2), ...
                            loc_layout(str_num,2)-(textbox(4)/2), [255, 255, 255]);
                    end
                else
                    target_on=false; p300_sti=0;
                end
            end
            
        end
        
        vbl = Screen('Flip',w);
        sti=vbl-prevVbl;
        p300_interval = p300_interval + (vbl-prevVbl);
        prevVbl = vbl;
        
        %% loop 종료 문
        if count_sequence==12
            n_sequence=n_sequence+1;
            count_sequence=0;
            toc
            tic
        end
        
        if n_sequence==10;
            n_sequence=0;
        end
        if count_run == cur_char*120
            run=false;
        end
    end

end

pause(1);
ppWrite(IO_ADD,222);

Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [255 255 255]);
Screen('Flip', w);
WaitSecs(2);
Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);

end

