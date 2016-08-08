function [ save_out ] = save_CFY(  CSP, LDA, band, time )
%VISUAL_ERD_ON Summary of this function goes here
%   Detailed explanation goes here
save_out=[];
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


play=true;
start=GetSecs;
ite=1;
while GetSecs < start+time
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
        
        if iscell(CSP)
            for i=1:length(CSP)
                tm=func_projection(fDat, CSP{i});
                ft=func_featureExtraction(tm, {'feature','logvar'});
                [cf_out(i)]=func_predict(ft, LDA{i})        
            end
            save_out(ite,:)=cf_out;
            ite=ite+1;

        else
            tm=func_projection(fDat, CSP);
            ft=func_featureExtraction(tm, {'feature','logvar'});
            [cf_out]=func_predict(ft, LDA);
        end
                
        
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
        %         pause(0.05);
    end
end

% fclose(fid);
end

