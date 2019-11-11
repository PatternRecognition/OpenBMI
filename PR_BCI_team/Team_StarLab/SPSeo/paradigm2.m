function [ output_args ] = paradigm2( varargin )

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttt_final = []; 
s_cell_final = []; 

%% MHLEE

Y_Dat = zeros(opt.total_run,37);
fileID1= fopen('save_cly4.dat','w');
fwrite(fileID1, Y_Dat,'double');
fclose(fileID1);
y_out2 = memmapfile('save_cly4.dat','Format',{'double' [15 72 37] 'x'} ,'Writable',true); 

f_order2 = memmapfile('save_order3.dat','Format',{'double' [15 72 6] 'x'},'Writable',true); 

notarestSP = memmapfile('save_notarestSP.dat','Format', {'double' [1 1] 'x'},'Writable',true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

s_idx = [1 7 13 19 25 31; 4 10 16 22 28 34; ...
    2 8 14 20 26 32; 5 11 17 23 29 35; ...
    3 9 15 21 27 33; 6 12 18 24 30 36]

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
notarestSP.Data.x = 2; 

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
s_cell = [];

for n_char = 1:length(copy_task) 
    ds_stop = false;  
    p_trig = c_trig; 
    lay_char = spell_char;
    layout = @spell_layout;
    chat_flag = true;
    % mh
    ds_mode = false;
    
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
    
    for n_seq = 1:nsequence 
        Screen('TextSize',w, ceil(text_size/2));
        for n_run=1:12       
            Screen('CopyWindow', offw, w);
            if online
                updated=true; 
                if updated
                    if n_seq == 1         
 
                        sti_Interval = 0.135;
                        pre_od = []; 
                        Draw_cell = order{n_seq}(n_run,:); 
                    else
                        sti_Interval = 0.065;
                        
                        if n_run == 1
                            while ~y_out2.Data.x(n_char, (n_seq-1)*12*n_run, 37) 
                                pause(0.3);
                                fprintf("waiting\n");
                            end                            
       
                            a = squeeze(y_out2.Data.x(n_char,:,:));
                            for k1 = 1:36
                                [aa aaa]=find(a(:,k1) ~= 0); 
                                L_idx(k1) = aa(end);
                                tm(k1) = y_out2.Data.x(n_char, aa(end), k1); 
                            end
                            
                        
                            if ~isempty(pre_od)                              
                                for k2=1:36
                                    if sum(k2==pre_od)
                                    else
                                        tm(k2) = nan;
                                    end
                                end
                                [tm_2 tm_2] = sort(tm, 'ascend');
                            else
                                 [tm_2 tm_2] = sort(tm, 'ascend'); 
                            end                           

                            %---------------------------- for ds   
                            tm5 = max(tm);
                            tm6 = abs(tm-tm5);
                            tm7 = tm6./sum(tm6(~isnan(tm))); 
                            

                            [tm8 tm9] = sort(tm7(~isnan(tm)), 'descend');
                            for k=1:length(tm8) 
                                tm11(k) = sum(tm8(1:k));
                            end
                                                  
                            [tm12 tm13] = find(tm11<0.7);   
%                               
                            pre_od = tm_2(tm13);
                       

                            tm_1 = zeros(1,36);
                            tm_1(1:length(pre_od)) = pre_od;
                            tm4 = reshape(tm_1', [6 6]);
                            s_order = tm4(s_idx);
                           
                            tm11 = [];                                  
                            ttt = squeeze(y_out2.Data.x(n_char, :,:));                 
                            %selection
                         
                            if tm8(1) > sum(tm8(2:end-1))                                
                                [tr1 tr2] = sort(tm, 'ascend');
                                spell_char{tr2(1)}
                                ds_stop = true;
                                pre_od = [];
                                sti_Interval = 0.135;
                                break;
                            end
                            
                            if sum(pre_od) == 0 || length(pre_od) == 0 
                                ds_stop = true;
                                break;
                            end
                           
                                
                            

                        end                        
                        if n_run>6 
                            Draw_cell = s_order(n_run-6,:);
                        else
                            Draw_cell = s_order(n_run,:);
                        end

                    end
                    
                else
                    Draw_cell = order{n_seq}(n_run,:);
                    if n_seq == 1
                    else
                        L_v = y_out2.Data.x(n_char, :,1);
                        L_v2 = find(L_v ~= 0);
                        tm = y_out2.Data.x(n_char, L_v2(end), :);
                        [tm2 tm3]= sort(tm);
                        s_str = [spell_char{tm3(1)} ' ' spell_char{tm3(2)} ' ' spell_char{tm3(3)} ' '  ...
                            spell_char{tm3(4)} ' ' spell_char{tm3(5)} ' ' spell_char{tm3(6)}];                        
                        s_num = [tm2(1) tm2(2) tm2(3) tm2(4) tm2(5) tm2(6)];

                        out_l(n_seq, :) = s_str;
                        out_v(n_seq, :) = s_num;
                    end
                    
                end
            else 
                Draw_cell = order{n_seq}(n_run,:);
            end
            

            

            s_cell = [s_cell; Draw_cell];
            [a1 b2] = size(s_cell);
            f_order2.Data.x(n_char, 1:a1,:) = s_cell;            
            
            Screen('TextSize',w, text_size);
            if p_trig == c_trig 
                for j = Draw_cell 
                    if j   
                        textbox = Screen('TextBounds', w, spell_char{j});
                        Screen('DrawText', w, spell_char{j}, loc_layout(j,1)-(textbox(3)/11*6), ...
                            loc_layout(j,2)-(textbox(4)/11*6), [255, 255, 255]);
                    else
                    end
                end
            else
                dstRects =  CenterRectOnPointd([0 0 80 80], loc_layout(Draw_cell,1), loc_layout(Draw_cell,2)); %seo,,, [a,b]=CenterRectOnPointd[450 100 500 1300]라고 예를 들면 a=475, b=700 나옴
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
        
        if ds_stop
            break;
        end
        
    end    
    
    Screen('CopyWindow', offw, w); 
    Screen('Flip', w);
    ttt_final = [ttt_final; ttt]; 
    s_cell_final = [s_cell_final; s_cell ; [999 999 999 999 999 999]]; 
    s_cell = [];
    WaitSecs(1);
    ppWrite(IO_ADD,14); 

end

save('tttfinal','ttt_final'); 
save('s_cellfinal','s_cell_final'); 
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