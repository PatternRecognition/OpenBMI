function vis_ERP_paradigm_ambulatory(trig, nSequence)

Screen('Preference', 'SkipSyncTests', 1);
% screen parameters
screenSize = 'full'; screenNum=2;


sti_Times=0.05; 
sti_Interval=0.135; 
timeRest = 2;

copy_text = 'KOREA_UNIVERSITY';
copy_task = [11, 15, 18, 5, 1, 36, 21, 14, 9, 22, 5, 18, 19, 9, 20, 25]; % KOREA_UNIVERSITY
order = importdata('C:\Users\cvpr\Desktop\삼성\ambulatory\paradigm\visual_ERP\func\random_order_v3.mat');

%% keys setting
startKey=KbName('space');
escapeKey = KbName('esc');
waitKey=KbName('`');

%% psychtoolbox
Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
black = BlackIndex(screenNum);

if ischar(screenSize) && strcmp(screenSize,'full')
    [w, rect] = Screen('OpenWindow', screenNum);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
else
    [w, rect] = Screen('OpenWindow', screenNum,[], screenSize);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
end

Screen('FillRect', w, black);
Textsize=50;%ceil(10);
Screen('TextSize',w,  ceil(10));
textbox = Screen('TextBounds', w, '+');

%% beep
freq_beep=22000;
beepLengthSecs=0.5;
[beep,samplingRate] = MakeBeep(500,beepLengthSecs,freq_beep);
Snd('Open');

%% send trigger
% brain vision setting
global IO_LIB IO_ADD;
IO_LIB=which('inpoutx64.dll');
IO_ADD=hex2dec('D010');

% openviber setting
tcp_ear = tcpclient('localhost', 15361);
padding=uint64(0);
timestamp=uint64(0);

% motion studio setting
s = serial('COM8');
set(s, 'BaudRate',9600,'DataBits',8,'parity','non','stopbits',1,'FlowControl','none');
fopen(s);

tr_pause = 111;
tr_resume = 222;
n_trig = trig(1);
t_trig = trig(2);


% ear trigger
stimulusPAUSE=[padding; uint64(tr_pause); timestamp];    % resume trigger
stimulusRESUME=[padding; uint64(tr_resume); timestamp];    % resume trigger

stim_n=[padding; uint64(trig(1)); timestamp];
stim_t=[padding; uint64(trig(2)); timestamp];


%%
spell_char = {'A','B','C','D','E','F',...
    'G','H','I','J','K','L',...
    'M','N','O','P','Q','R',...
    'S','T','U','V','W','X',...
    'Y','Z','1','2','3','4',...
    '5','6','7','8','9','_'};
speller_size = [6 6];
lay_char = spell_char;
layout = @spell_layout;

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

dot = false;

%% paradigm start
prevVbl = Screen('Flip',w);
disp('Press space to start Visual ERP experiment ')
[ keyIsDown, ~, keyCode ] = KbCheck;
while ~keyCode(startKey)
    Screen('TextSize',w, Textsize);
    DrawFormattedText(w, 'Press space to start Visual ERP experiment ', 'center', 'center', [255 255 255]);
    [ keyIsDown, ~, keyCode ] = KbCheck;
    Screen('Flip', w);
end
disp('Visual Speller task will start in 3 secs')
Screen('TextSize',w, Textsize);
DrawFormattedText(w,'Visual Speller task will start in 3 secs','center','center',[255 255 255]);
Screen('Flip', w);
WaitSecs(1);
Screen('FillRect', w, black);
WaitSecs(2);
Screen('TextSize', w, ceil(10));

for n_char = 1:length(copy_task) %%korea university 부분
     %% Pause 실험 반 하고 좀 쉬어야지
     if (rem(n_char,16)==1) 
        write(tcp_ear, stimulusPAUSE);
        ppWrite(IO_ADD,tr_pause);
        fwrite(s,uint32(tr_pause),'uint32');
        if (n_char~=1)
            [ keyIsDown, ~, keyCode ] = KbCheck;
            while ~keyCode(startKey)
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w,'Break Time\n\nPress space to resume the experiment','center','center',[255 255 255]);
                Screen('Flip', w);
                [ keyIsDown, ~, keyCode ] = KbCheck;
            end
            
            Screen('TextSize',w, Textsize);
            DrawFormattedText(w,'Visual ERP task will resume in 3 secs','center','center',[255 255 255]);
            Screen('Flip', w);
            WaitSecs(3);
        end
        write(tcp_ear, stimulusRESUME);
        ppWrite(IO_ADD,tr_resume);
        fwrite(s,uint32(tr_resume),'uint32');
    end
    fprintf('Trial #%.d \t Target %.d \n',n_char, copy_task(n_char));


    %% timer
    Screen('FillRect', offw, [0 0 0]);
    layout(offw, copy_task, lay_char, loc_layout, text_size, rect, dot);
    
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
    WaitSecs(2);
    target_ind=copy_task(n_char); %% target highlight

        Screen('TextSize',w, text_size);
        textbox = Screen('TextBounds', w, spell_char{target_ind});
        Screen('DrawText', w, spell_char{target_ind}, loc_layout(target_ind,1)-(textbox(3)/2), ...
            loc_layout(target_ind,2)-(textbox(4)/2), [255, 255, 255]);

    
    Screen('Flip', w);
    WaitSecs(0.5);
    Screen('CopyWindow', offw, w);
    Screen('Flip', w);
%     ppWrite(IO_ADD,p_trig);  % 15 start
    WaitSecs(2);
    
    for  n_seq = 1:nSequence %nsequence 만큼 하나의 target character를 반복
        Screen('TextSize',w, ceil(text_size/2));
        for n_run=1:12       %run 6X6 speller
            Screen('CopyWindow', offw, w);
            Draw_cell = order{n_seq}(n_run,:);
            
            Screen('TextSize',w, text_size);
            for j = Draw_cell %A presentation in a run
                textbox = Screen('TextBounds', w, spell_char{j});
                Screen('DrawText', w, spell_char{j}, loc_layout(j,1)-(textbox(3)/11*6), ...
                    loc_layout(j,2)-(textbox(4)/11*6), [255, 255, 255]);
            end
                        
            vbl = Screen('Flip', w, 0, 1);
            %% trigger
            trig = ismember(Draw_cell, target_ind);  %% 타겟부분일경우만 trigger
            if sum(trig)      %target
                write(tcp_ear, stim_t)
                ppWrite(IO_ADD,t_trig);
                fwrite(s,uint32(t_trig),'uint32');
            else            %non-target
                write(tcp_ear, stim_n)
                ppWrite(IO_ADD,n_trig);
                fwrite(s,uint32(n_trig),'uint32');
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
%     ppWrite(IO_ADD,14); %online 파일의 c_"n초기화->eog switch가 안들어오도록

    tic;
    while toc < timeRest
        [ keyIsDown, ~, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(2);
                Screen('CloseAll');
                ShowCursor;
                fclose('all');
                Priority(0);
                return;
            elseif keyCode(waitKey)
                write(tcp_ear, stimulusPAUSE);
                ppWrite(IO_ADD,tr_pause);
                fwrite(s,uint32(tr_pause),'uint32');
                while ~keyCode(startKey)
                    Screen('FillRect', w, black);
                    [ keyIsDown, ~, keyCode ] = KbCheck;
                    Screen('TextSize',w, Textsize);
                    DrawFormattedText(w, 'Pause experiment', 'center', 'center', [255 255 255]);
                    Screen('Flip', w);
                end
                Screen('TextSize',w, Textsize);
                DrawFormattedText(w,'Visual ERP task will resume in 3 secs','center','center',[255 255 255]);
                Screen('Flip', w);
                WaitSecs(3);
                
                write(tcp_ear, stimulusRESUME);
                ppWrite(IO_ADD,tr_resume);
                fwrite(s,uint32(tr_resume),'uint32');
            end
        end
    end
    
end

%%
pause(1);
sca;
fclose('all');

output_args = 'Done all process...';

end







