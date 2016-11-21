function Makeparadigm_SSVEP (varargin)
% Makeparadigm_SSVEP (Experimental paradigm):
% 
% Description:
%   Basic SSVEP experiment paradigm using psychtoolbox.
%   It shows flickering boxes. Before flickering, random target is indicated in yellow.
% 
% Example:
%   Makeparadigm_SSVEP ({'time_sti',5;'num_trial',10;'time_rest',3;'freq',[7.5 10 12 15 20];'boxSize',150;'betweenBox',200});
% 
% Input: (Nx2 size, cell-type)
%   time_sti   - time for a stimulus [s]
%   time_rest  - time for rest, showing nothing but black screen [s]
%   color      - RGB color of stimulus box (e.g.[255 255 255])
%   num_trial  - number of trials per class
%   freq       - number of classes you want, from 1 to 5 (up,left,center,right,down)
%   boxSize    - size of stimulus box [pixel]
%   betweenBox - distance between stimulus boxes [pixel]
%   num_screen  - number of screen to show 
%   screen_size - size of window showing experimental stimulus
%                 'full' or matrix (e.g.[0 0 300 300])
% 


%% variables
in=opt_cellToStruct(varargin{:});
f=in.freq;
if isfield(in,'color'),color = in.color;
else color=[255 255 255];end
n1=0;n2=0;n3=0;n4=0;n5=0;
escapeKey = KbName('esc');
waitKey=KbName('s');

%% trigger
global IO_ADDR IO_LIB;
IO_ADDR=hex2dec('D010');
IO_LIB=which('inpoutx64.dll');

%% psychtoolbox
% screenNum = 2;
if ~isfield(opt,'num_screen'),num_screen=max(screens); else num_screen=opt.num_screen; end
if ~isfield(opt,'screen_size'),screen_size='full'; else screen_size=opt.screen_size; end
black = BlackIndex(w);
if strcmp(screen_size,'full')
    [w, wRect]=Screen('OpenWindow',screenNumber, black);
else
    [w, wRect]=Screen('OpenWindow',screenNumber, black, screen_size);
end
% [w, rect] = Screen('OpenWindow', num_screen,[0 0 1680 1050]);
Screen('FillRect', w, black);

%% frequency
len_f = length(f);
ssvep_interval = zeros(1,len_f);
isi_ssvep = 1./f;

%% stimuli
% Order of stimuli (eyes movement)
% idx = randperm(in.num_trial);
order_task = repmat(1:len_f,1,in.num_trial);
% order_task = order_task(idx);
order_task=Shuffle(order_task);

%% box(stimulus) position
%           1
%      2    3    4
%           5
s_size=in.boxSize; wd=rect(3); h=rect(4); inter=in.betweenBox;
x1=[wd/2-s_size/2, wd/2-3*s_size/2-inter, wd/2-s_size/2, wd/2+s_size/2+inter, wd/2-s_size/2];
x2=[wd/2+s_size/2, wd/2-s_size/2-inter, wd/2+s_size/2, wd/2+3*s_size/2+inter, wd/2+s_size/2];
y1=[h/2-3*s_size/2-inter, h/2-s_size/2, h/2-s_size/2, h/2-s_size/2, h/2+s_size/2+inter];
y2=[h/2-s_size/2-inter, h/2+s_size/2, h/2+s_size/2, h/2+s_size/2, h/2+3*s_size/2+inter];


%% start
prevVbl = Screen('Flip',w);
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Click to start an experiment', 'center', 'center', [255 255 255]);
Screen('Flip', w);
GetClicks(w);
ppTrigger(111);


for t=1:length(order_task)
    
    %% show cue
    % background
    for j=1:len_f
        Screen('FillRect', w, color, [x1(j),y1(j),x2(j),y2(j)]);
    end
    % cue of this trial
    switch order_task(t)
        case 1
            Screen('FillRect', w, [255 255 0], [x1(1),y1(1),x2(1),y2(1)]);
            Screen('Flip',w);
            WaitSecs(2);
            ppTrigger(1)
            n1=n1+1;
        case 2
            Screen('FillRect', w, [255 255 0], [x1(2),y1(2),x2(2),y2(2)]);
            Screen('Flip',w);
            WaitSecs(2);
            ppTrigger(2)
            n2=n2+1;
        case 3
            Screen('FillRect', w, [255 255 0], [x1(3),y1(3),x2(3),y2(3)]);
            Screen('Flip',w);
            WaitSecs(2);
            ppTrigger(3)
            n3=n3+1;
        case 4
            Screen('FillRect', w, [255 255 0], [x1(4),y1(4),x2(4),y2(4)]);
            Screen('Flip',w);
            WaitSecs(2);
            ppTrigger(4)
            n4=n4+1;
        case 5
            Screen('FillRect', w, [255 255 0], [x1(5),y1(5),x2(5),y2(5)]);
            Screen('Flip',w);
            WaitSecs(2);
            ppTrigger(5)
            n5=n5+1;
    end
    
    % stimuli
    s=0;
    tic
    %         for i=1:len_f
%     if f
        %                 run=true;
        while s<in.time_sti % && run
            [ keyIsDown, seconds, keyCode ] = KbCheck;
            if keyIsDown
                if keyCode(escapeKey)
                DrawFormattedText(w, 'End of experiment', 'center', 'center', [255 255 255]);
                Screen('Flip', w);
                WaitSecs(1);
                Screen('CloseAll');
                fclose('all');
                return
                end
            elseif keyCode(waitKey)
                GetClicks(w);
            end
            
            
            for n=1:length(f)
                if(ssvep_interval(n)>= isi_ssvep(n))
                    Screen('FillRect', w, color, [x1(n),y1(n),x2(n),y2(n)]);
                    ssvep_interval(n) = 0;
                end
            end
            
            vbl = Screen('Flip',w);
            ssvep_interval = ssvep_interval + (vbl-prevVbl);
            prevVbl = vbl;
            s=toc;
        end
%     end
    %         end
    Screen('Flip', w);
    WaitSecs(in.time_rest);
    
    disp(sprintf('Trial #%.0d',t))
    disp(sprintf('# of trials per class\n1: %.0d\t,\t2: %.0d\t,\t3: %.0d\t,\t4: %.0d\t,\t5: %.0d',n1,n2,n3,n4,n5))

end

%% End
Screen('TextSize',w, 50);
DrawFormattedText(w, 'Thank you', 'center', 'center', [255 255 255]);
Screen('Flip', w);
WaitSecs(2);

% End trigger
ppTrigger(222);

Screen('CloseAll');
ShowCursor;
fclose('all');
Priority(0);
