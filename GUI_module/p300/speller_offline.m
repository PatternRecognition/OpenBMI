function [ output_args ] = speller_offline( speller_type, varargin )

opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'port'), error('No input port information'), end
if ~isfield(opt,'screenSize'),screenSize='full';
else screenSize=[0,0,opt.screenSize];end
if ~isfield(opt,'text'),test_character='DEFAULT_TEXT';
else test_character=upper(opt.text);end
if ~isfield(opt,'nSequence'),opt.nSequence=10;end
if ~isfield(opt,'screenNum'),opt.screenNum=2;end
if ~isfield(opt,'sti_Times'),opt.sti_Times=0.135;end
if ~isfield(opt,'sti_Interval'),opt.sti_Interval=0.05;end
if ~isfield(opt,'trigger'),opt.trigger=[1,2];end

%% trigger
global IO_LIB;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec(opt.port);
% IO_ADD=hex2dec('C010');

%% OFFLINE speller
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

spell_num=[ 1,  2,  3,  4,  5,  6;
    7,  8,  9, 10, 11, 12;
    13, 14, 15, 16, 17, 18;
    19, 20, 21, 22, 23, 24;
    25, 26, 27, 28, 29, 30;
    31, 32, 33, 34, 35, 36;
    1,  7,  13, 19, 25, 31;
    2,  8,  14, 20, 26, 32;
    3,  9,  15, 21, 27, 33;
    4,  10, 16, 22, 28, 34;
    5,  11, 17, 23, 29, 35;
    6,  12, 18, 24, 30, 36];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
speller_size = [6 6];
escapeKey = KbName('esc');
waitKey=KbName('space');

%% flickering_order
if strcmp(speller_type,'random')
    load cell_order_new; % 일단. 10 seq
elseif strcmp(speller_type,'RC')
    load cell_order_new
%     cell_order=num2cell(opt.nSequence*ones(1,opt.nSequence),[1,opt.nSequence]);
%     cell_order=cellfun(@make_RC_order,cell_order);
elseif strcmp(speller_type,'face')
    load cell_order_new
else
    error('OpenBMI: Incorrect speller type')
end

%% psychtoolbox setting
if ischar(screenSize) && strcmp(screenSize,'full')
    Screen('Preference', 'SkipSyncTests', 1);       % 주의
    [w, rect] = Screen('OpenWindow', opt.screenNum );
else
    Screen('Preference', 'SkipSyncTests', 1);       % 주의
    [w, rect] = Screen('OpenWindow', opt.screenNum,[], screenSize);
    % [w, rect] = Screen('OpenWindow', screenNum,[], [0 0 1280 720]);
end

loc_layout = proc_getlayout(speller_size, rect);
black = BlackIndex(w);
Screen('FillRect', w, black);

t_text_size = 60;  % target text size (big char. in center)
n_text_size = 50;  % 'brain-computer-interface'

sti_Times = 0.135;%per/S 10 %%여기 다시한번 체크할것
sti_Interval = 0.05;%per/S

vbl = Screen(w, 'Flip');
ifi = Screen('GetFlipInterval', w);

normal_layout(w, test_character, spell_char, loc_layout, n_text_size);

for n_char = 1:length(test_character)
    target_ind = find(test_character(n_char) == spell_char); %find the positions for target
    %% timer
    normal_layout(w, test_character, spell_char, loc_layout, n_text_size);
    Screen('DrawText', w, test_character(1:n_char-1), 10, 10+n_text_size/2, [255, 255, 255]);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, ceil(t_text_size));
    Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
    
    Screen('Flip', w);
    pause(2);
    
    i2=strcmpi(spell_char2,test_character(n_char)); %% target highlight
    Screen('TextSize',w, n_text_size);
    textbox = Screen('TextBounds', w, spell_char(i2));
    Screen('DrawText', w, spell_char(i2), loc_layout(i2,1)-(textbox(3)/2), ...
        loc_layout(i2,2)-(textbox(4)/2), [200, 200, 200]);
    Screen('Flip', w);
    pause(0.5);
    
    normal_layout(w, test_character, spell_char, loc_layout, n_text_size);
    Screen('DrawText', w, test_character(1:n_char-1), 10, 10+n_text_size/2, [255, 255, 255]);
    Screen('TextFont',w, 'Arial');
    Screen('TextStyle', w, 0);
    Screen('TextSize',w, ceil(t_text_size));
    Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
    
    start_1 = Screen('Flip', w);
    pause(2);
    
    for n_seq = 1:opt.nSequence %nsequence 만큼 하나의 target character를 반복
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                Screen('CloseAll');
                fclose('all');
            elseif keyCode(waitKey)
                GetClicks(w);
            end
        end
        
        for n_run=1:sum(speller_size)       %run 6X6 speller
            
            Screen('TextFont',w, 'Arial');
            Screen('TextSize',w, ceil(n_text_size/2.5));
            Screen('TextStyle', w, 0);
            Screen('DrawText', w, test_character, 10, 10, [255, 255, 255]);
            Screen('DrawText', w, test_character(1:n_char-1), 10, 10+n_text_size/2, [255, 255, 255]);
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
            trig = ismember(Draw_cell, target_ind);
            if sum(trig)    % target
                ppWrite(IO_ADD,opt.trigger(1));
            else            % non-target
                ppWrite(IO_ADD,opt.trigger(2));
            end
            
            start_2 = Screen('Flip', w, start_1 + sti_Times);
            normal_layout(w, test_character, spell_char, loc_layout, n_text_size);
            Screen('TextSize',w, ceil(n_text_size/2.5));
            Screen('DrawText', w, test_character(1:n_char-1), 10, 10+n_text_size/2, [255, 255, 255]);
            Screen('TextFont',w, 'Arial');
            Screen('TextStyle', w, 0);
            Screen('TextSize',w, ceil(t_text_size));
            Screen('DrawText', w, test_character(n_char), (rect(3)/2)-(ceil(t_text_size)/2), 45, [255, 255, 255]);
            start_3 = Screen('Flip', w, start_2 + sti_Interval);
            start_1 = start_3;
        end
    end
    normal_layout(w, test_character, spell_char, loc_layout, n_text_size);
    WaitSecs(2);
    Screen('Flip', w);
    
end
pause(1);
ppWrite(IO_ADD,20); %15=end
Screen('CloseAll');
fclose('all');
end

% function make_RC_order(cell)
% end