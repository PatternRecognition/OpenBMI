function [  ] = visual_CFY_on(  CSP, LDA, band, varargin )
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


play=true;
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
        
        if iscell(CSP)
            for i=1:length(CSP)
                tm=func_projection(fDat, CSP{i});
                ft=func_featureExtraction(tm, {'feature','logvar'});
                [cf_out(i)]=func_predict(ft, LDA{i})
                
            end
            if ~isempty(CLY_LDA)
                for i2=1:length(CLY_LDA)
                    [de_out(i2)]=func_predict(cf_out, CLY_LDA{i2});
                end
                [out b]=min(de_out');
                b
    
            end
            %                               str=sprintf('1: %d 2: %d 3: %d 4: %d 5: %d 6: %d 7: %d 8: %d 9: %d 10: %d', ...
%                     cf_out(1),cf_out(2),cf_out(3),cf_out(4),cf_out(5),cf_out(6),cf_out(7),cf_out(8),cf_out(9),cf_out(10));
%             fprintf(fid, str);
            

%                 str
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
%                 pause(0.05);
    end
end

% fclose(fid);
end

