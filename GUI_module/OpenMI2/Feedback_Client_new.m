function [ output_args ] = Feedback_Client_new( CSP, LDA, band,fs,t_stimulus,hh,hh1, varargin )

opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'channel')
    error('OpenBMI: Channels are not selected.')
end

if strcmp(opt.TCPIP, 'on')
    d = tcpip('localhost', 3000, 'NetworkRole', 'Client');
    set(d, 'OutputBufferSize', 1024); % Set size of receiving buffer, if needed.
    
    % Trying to open a connection to the server.
    while(1)
        try
            fopen(d);
            break;
        catch
            fprintf('%s \n','Cant find Server');
        end
    end
    connectionSend = d;
end

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
buffer_size=opt.buffer_size; % #point -> ms
data_size=opt.data_size;
feedback_t=opt.feedback_freq; % feedback frequency
Dat=zeros(buffer_size, size(state.chan_sel,2));
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));
escapeKey = KbName('esc');
waitKey=KbName('s');
cf_out=[];

rcf_out=[0 0 0];
a=0;
t=[];
midx=1;
% mmtime2=0;
while true
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            return
        elseif keyCode(waitKey)
            warning('stop')
            GetClicks([]);
        end
    end
    flushoutput(d)
    
    
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
%     markerdescr=round(markerdescr);
    if markerdescr==111 % 시작
        tic;
    end
    if ~isempty(markerdescr)
        if (0<markerdescr)&&(markerdescr<4)
            mmarker(midx)=markerdescr;
        end
    end
    if markerdescr==222 % 종료
        SAVE_DATA.marker=mmarker;
        SAVE_DATA.x=orig_Dat(buffer_size+1:end,:);
        SAVE_DATA.marker_time=t;
        
        for kk=1:length(t)
            SMT(:,:,kk)=orig_Dat(t(kk):t(kk)+t_stimulus/1000*state.fs-1,:);
        end
        SAVE_DATA.SMT=SMT;
        SAVE_DATA.fs=state.fs;
        
        c = clock;
        direct=fileparts(which('OpenBMI'));
        save(fullfile(direct, 'log',sprintf('%d%02d%02d_%02d.%02d_MotorImagery.mat',c(1:5))),'SAVE_DATA');
        bbci_acquire_bv('close');
        fclose(d);
        fclose('all');
        a=1;
        break;
    end
    if ~isempty(markerdescr)
        if (markerdescr==1)||(markerdescr==2)||(markerdescr==3)
            mmarker(midx)=markerdescr;
            t(midx)=ceil(size(orig_Dat,1)+markertime/1000*state.fs);
            midx=midx+1;
        end
    end
    
    orig_Dat=[orig_Dat; data];
    if length(orig_Dat)>buffer_size*state.fs/1000 % prevent overflow
        Dat=orig_Dat(end-buffer_size*state.fs/1000+1:end,:);
        Dat2.x=Dat;
        Dat2.fs=state.fs;
        Dat=prep_resample(Dat2,fs);
        fDat=prep_filter(Dat, {'frequency', band;'fs',fs});%1000});%state.fs });
        fDat = fDat.x(:,opt.channel);
        fDat=fDat(end-data_size*fs/1000+1:end,:); % data
        if iscell(CSP)
            for i=1:size(CSP,1)
                tm=func_projection(fDat, CSP{i,1});
                ft=func_featureExtraction(tm, {'feature','logvar'});
                [cf_out(i)]=func_predict(ft, LDA{i,1})
            end
        else
            tm=func_projection(fDat, CSP);
            ft=func_featureExtraction(tm, {'feature','logvar'});
            [cf_out]=func_predict(ft, LDA)
        end
    end
    
    bar(hh,cf_out);
    axis(hh,[0 4 -20 20])
    drawnow
    
    if strcmp(opt.TCPIP, 'on')
        flushoutput(d)
        fwrite(d,cf_out,'double')
    end
    if ~isempty(t)
%     if (markerdescr==1)||(markerdescr==2)||(markerdescr==3)
        if t(midx-1)+t_stimulus/1000*state.fs<length(orig_Dat)&&length(orig_Dat)<t(midx-1)+t_stimulus/1000*state.fs+20 % 문제가 만타... 이걸어떻게 해줘야할까...
            ddat=orig_Dat(end-t_stimulus/1000*state.fs:end,:);
            if iscell(CSP)
                for i=1:size(CSP,1)
                    rtm=func_projection(ddat, CSP{i,1});
                    rft=func_featureExtraction(rtm, {'feature','logvar'});
                    [rcf_out(i)]=func_predict(rft, LDA{i,1});
                end
            else
                rtm=func_projection(ddat, CSP);
                rft=func_featureExtraction(rtm, {'feature','logvar'});
                [rcf_out]=func_predict(rft, LDA);
            end
            [~,b]=max(abs(rcf_out));
            axes(hh1);
            if b==1
                imshow('right.jpg');
            elseif b==2
                imshow('left.jpg');
            else
                imshow('down.jpg');
            end
            drawnow
        end
%     end
    end


    %
    % if length(orig_Dat)>buffer_size*state.fs/1000
    %     ddat=orig_Dat(end-749:end,:);
    %     if iscell(CSP)
    %         for i=1:length(CSP)
    %             rtm=func_projection(ddat, CSP{i,1});
    %             rft=func_featureExtraction(rtm, {'feature','logvar'});
    %             [rcf_out(i)]=func_predict(rft, LDA{i,1});
    %         end
    %     else
    %         rtm=func_projection(ddat, CSP);
    %         rft=func_featureExtraction(rtm, {'feature','logvar'});
    %         [rcf_out]=func_predict(rft, LDA);
    %     end
    % end
    % [~,b]=max(abs(rcf_out));
    % if (mmtime2+700<length(orig_Dat))&&(length(orig_Dat)<mmtime2+800)
    %     axes(hh1);
    %     if b==1
    %         imshow('right.jpg');
    %     elseif b==2
    %         imshow('left.jpg');
    %     else
    %         imshow('down.jpg');
    %     end
    %     drawnow
    % end
end

