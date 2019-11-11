function [ output_args ] = paradigm1( varargin )

%% Init
opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'port'), error('No input port information');
else port= opt.port;end

if ~isfield(opt,'screenSize')
    screenSize='full';
elseif ischar(opt.screenSize)
    screenSize = opt.screenSize;
else
    screenSize=[0,0,opt.screenSize];
end
if ~isfield(opt,'text')
    copy_task=1;
else
    copy_task=opt.text;
end

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
top4syminthisseq = cell(1,nSequence); 


order_myData = zeros(12,6);
fileID1= fopen('order_saved.dat','w');
fwrite(fileID1,order_myData,'double'); 
fclose(fileID1);
m1 = memmapfile('order_saved.dat','Format',{'double' [12 6] 'x'},'Writable',true);


%% check for connection
if online 
    sock = tcpip('localhost', 30000, 'NetworkRole', 'Server'); 
    set(sock, 'InputBufferSize', 1024); 
    set(sock, 'OutputBufferSize', 1024);  
    % Open connection to the client
    fprintf('%s \n','Client Connecting...');
    fopen(sock); 
    fprintf('%s \n','Client Connected');
    set(sock,'Timeout', 1.5);
end

order = importdata('C:\Users\cvpr\Desktop\Experiments\experiment_main\session3_files\random_order_v3_sp190730.mat');
order_origin = order; 
nsequence = nSequence; 
T_char=[];  
%%
if ischar(screenSize) && strcmp(screenSize,'full')
    [w, rect] = Screen('OpenWindow', screenNum);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
else
    [w, rect] = Screen('OpenWindow', screenNum,[], screenSize);
    [offw] = Screen('OpenOffscreenWindow', -1, [0 0 0], rect);
end

%%

spell_char = {'A','B','C','D','E','F',...
    'G','H','I','J','K','L',...
    'M','N','O','P','Q','R',...
    'S','T','U','V','W','X',...
    'Y','Z','1','2','3','4',...
    '5','6','7','8','9','-'};
img_folder = 'C:\Users\cvpr\Desktop\Experiments\experiment_main\session3_files\PNG';
imgs_name = arrayfun(@(x) sprintf('%02d_off.png', x), 1:36, 'Uni', false);
imgs = cellfun(@(x) imread(fullfile(img_folder, x)), imgs_name, 'Uni', false);
images(:,1) = cellfun(@(x) Screen('MakeTexture', w, x), imgs); 
imgs_name = arrayfun(@(x) sprintf('%02d_on.png', x), 1:36, 'Uni', false);
imgs = cellfun(@(x) imread(fullfile(img_folder, x)), imgs_name, 'Uni', false);

images(:,2) = cellfun(@(x) Screen('MakeTexture', w, x), imgs); 
clear imgs imgs_name

speller_size = [6 6];
escapeKey = KbName('esc');
waitKey=KbName('*');

chat_flag = false;
lay_char = images;
layout = @image_layout;
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
dot = false;
time_delay = 0;

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
WaitSecs(1);
Screen('Flip', w);

%% Beep
beepLengthSecs=0.1;
rate=44100;
beepY = MakeBeep(freq,beepLengthSecs,rate).*0.2;
Snd('Open'); 
%% Hybrid order
hybOrd = repmat([1 2], 1, ceil(length(copy_task)/2));
if ~opt.online 
    hybOrd = sort(hybOrd); 
end
%% Start Paradgims
Snd('Play',beepY,rate); 
pause(1);

%%
h_trig = 15;
c_trig = 16;
p_trig = 15;
n_tri = 1; 

t_tri = 11;
p_tri = 2; 
a_tri = 3; 

result= 0;
%% Start
ppWrite(IO_ADD, 111);
for n_char = 1:length(copy_task) 
    
    p_trig = c_trig; %seo,,, 15¿¡ 16Áý¾î³ÖÀ½ °á±¹ 16µÊ,,, ÀÌÁþ ¿ÖÇÏÂ¡   
    lay_char = spell_char;
    layout = @spell_layout;
    chat_flag = true;


    %% timer
    Screen('FillRect', offw, [0 0 0]); 
    layout(offw, copy_task, lay_char, loc_layout, text_size, rect, dot); 
    
    if online && chat_flag 
        if~isempty(T_char) 
            Screen('TextSize',offw, ceil(text_size/3));
            Screen('DrawText', offw, T_char,0, 0, [255, 255, 255]);
        end
    end
    Screen('CopyWindow', offw, w); 
    Screen('Flip', w); 
    WaitSecs(1);
    target_ind=copy_task(n_char); 

    if p_trig == c_trig 
        Screen('TextSize',w, text_size);
        textbox = Screen('TextBounds', w, spell_char{target_ind});
        Screen('DrawText', w, spell_char{target_ind}, loc_layout(target_ind,1)-(textbox(3)/2), ...
            loc_layout(target_ind,2)-(textbox(4)/2), [255, 255, 255]);
    else
        Screen('TextSize',w, text_size);
        dstRect =  CenterRectOnPointd([0 0 80 80], loc_layout(target_ind,1), loc_layout(target_ind,2));
        dstRect = dstRect - (10/11*6);
        Screen('DrawTexture', w, images(target_ind,2), [],  dstRect); 
    end
    
    Screen('Flip', w); 
    Snd('Play',beepY, rate);
    WaitSecs(0.5);
    Screen('CopyWindow', offw, w); 
    Screen('Flip', w);
    ppWrite(IO_ADD,p_trig);  
    WaitSecs(1);
    
    for  n_seq = 1:nsequence 
        Screen('TextSize',w, ceil(text_size/2));
        if online
            switch n_seq
                case {1,2}
                    order = order_origin;
                case {3,4,5,6,7,8,9,10} 
                    order_maker_sp();
                    order_n_seq = ans{n_seq}; 
                    m1.Data.x = order_n_seq;
                    CumulativeOrder{n_char} = m1.Data.x; 
            end
        end

        for n_run=1:12       
            Screen('CopyWindow', offw, w);
            if online
                if n_seq < 3
                    Draw_cell = order{n_seq}(n_run,:);
                else %n_seq 3,4,5,6...
                    Draw_cell = m1.Data.x(n_run,:);
                end
            else
                Draw_cell = order{n_seq}(n_run,:);
            end
       
      





           
            Screen('TextSize',w, text_size);
            if p_trig == c_trig 
                for j = Draw_cell 
                    textbox = Screen('TextBounds', w, spell_char{j});
                    Screen('DrawText', w, spell_char{j}, loc_layout(j,1)-(textbox(3)/11*6), ...
                        loc_layout(j,2)-(textbox(4)/11*6), [255, 255, 255]);
                end
            else
                dstRects =  CenterRectOnPointd([0 0 80 80], loc_layout(Draw_cell,1), loc_layout(Draw_cell,2)); 
                dstRects = dstRects - (10/11*6);
                Screen('DrawTextures', w, images(Draw_cell,2), [],  dstRects');
            end
            
            vbl = Screen('Flip', w, 0, 1); 
    
            %% trigger
            trig = ismember(Draw_cell, target_ind); 
            if sum(trig)      
                ppWrite(IO_ADD,t_tri); 
            else            
                ppWrite(IO_ADD,n_tri); 
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
    ppWrite(IO_ADD,14); 
    
    if online 
        tic;
        result = fread(sock, 1); 
        time_delay = toc;
        if ~isempty(result)
            fid = fopen('C:\Users\cvpr\Documents\Demo2\Assets\aaa.txt', 'w'); 
            fwrite(fid, string(result));
            fclose(fid);
            if chat_flag 
                if any(result == [37])
                    T_char = [];
                else
                    str = spell_char{result};
                    T_char=[T_char str];
                    clear str
                end
            end
        end
    else
        result = copy_task(n_char); 
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
                ppWrite(IO_ADD,20); 
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

CumulativeOrder{1} = order_origin{1};
CumulativeOrder{2} = order_origin{2};
save('SavedCumulativeOrder','CumulativeOrder')
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

ppWrite(IO_ADD,222); 
GetClicks(w);
pause(1);


sca; 
fclose('all');

output_args = 'Done all process...';

end