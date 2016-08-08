function [  ] = visual_CFY_on2(  CSP, LDA, band, varargin )
%VISUAL_ERD_ON Summary of this function goes here
%   Detailed explanation goes here

if ~isempty(varargin)
    CLY_LDA=varargin{:};
else
    CLY_LDA=[];
end

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
orig_Dat=[];

buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));

escapeKey = KbName('esc');
waitKey=KbName('s');
%% test
% fid = fopen('ny.txt','wt');

%% image load
img_right=imread('\Stimulus\right_arrow.jpg');
img_left=imread('\Stimulus\\left_arrow.jpg');
img_foot=imread('\Stimulus\up_arrow.jpg');
img_rest=imread('\Stimulus\rest_square.jpg');

screenNumber=1;
gray=GrayIndex(screenNumber);
% screenRes = [0 0 640 480];
% [w, wRect]=Screen('OpenWindow',screenNumber, gray, screenRes);
[w, wRect]=Screen('OpenWindow', 2, gray);

play=true;
temp=0;
play=true;
buffer=[];
buffer2=[];
bu_i=1;
tic
stop=true;
temp=1;
while play
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    orig_Dat=[orig_Dat; data];
    if length(orig_Dat)>buffer_size % prevent overflow
        Dat=orig_Dat(end-buffer_size+1:end,:);
        orig_Dat=Dat;  %%
        Dat2.x=Dat;
        Dat2.fs=state.fs;
        %         Dat=prep_resample(Dat2,500);
        Dat=Dat2.x;
        fDat=prep_filter(Dat, {'frequency', band;'fs',1000});%state.fs });
        fDat=fDat(end-data_size:end,:); % data
        
        
        for i=1:length(CSP)
            tm=func_projection(fDat, CSP{i});
            ft=func_featureExtraction(tm, {'feature','logvar'});
            [cf_out(i)]=func_predict(ft, LDA{i}) 
        end
        %% OVR strategy
%         if cf_out(10)<3
%             output=4;
%         else
%             [a b]=min(cf_out(7:9));
%             output=b;
%         end
%         
        
        if cf_out(10)<3
            b=4;
        elseif cf_out(9)<1
            b=3
        else
            if cf_out(1)-3<0
                b=1;
            else
                b=2;
            end           

        end
        
        buffer(bu_i)=b;
        bu_i=bu_i+1;
        if length(buffer)>10
            tm_bf=buffer(end-8:end);
            if length(find(tm_bf==4))>3
                b=4;
            end
            if length(find(tm_bf==4))==0
                if temp==4
                    b=3;
                    temp=0;
                else
                    temp=temp+1;
                end
                
            end
        end
        
        
        switch b
            case 1 %right class
                image=img_right;
            case 2 %left class
                image=img_left;
            case 3 %foot class
                image=img_foot;
            case 4
                image=img_rest;
        end
          tex1=Screen('MakeTexture', w, image );
            Screen('DrawTexture', w, tex1);    
             [VBLTimestamp startrt]=Screen('Flip', w);
        
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                ShowCursor;
                play=false;
            elseif keyCode(waitKey)
                warning('stop')
                GetClicks(w);
                Screen('Close',tex1);
            else
            end
        end
                        pause(0.05);
    end
end

% fclose(fid);
end

