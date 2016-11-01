function [ output_args ] = Feedback_Client( CSP, LDA, band, varargin )

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
buffer_size=opt.buffer_size;
data_size=opt.data_size;
feedback_t=opt.feedback_freq; % feedback frequency
Dat=zeros(buffer_size, size(state.chan_sel,2));
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));
escapeKey = KbName('esc');
waitKey=KbName('s');
a=1;
cf_out=[];
tic
while true
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            
        elseif keyCode(waitKey)
            
        end
    end
    
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    orig_Dat=[orig_Dat; data];
    if length(orig_Dat)>buffer_size % prevent overflow        
        Dat=orig_Dat(end-buffer_size+1:end,:);        
        Dat2.x=Dat;
        Dat2.fs=state.fs;
        Dat=prep_resample(Dat2,500);
        Dat=Dat.x;
        %     if toc>feedback_t
        a=a+1;
        fDat=prep_filter(Dat, {'frequency', band;'fs',500});%state.fs });
        fDat = fDat(:,opt.channel);
        
        fDat=fDat(end-data_size:end,:); % data
        if iscell(CSP)
            for i=1:length(CSP)
                tm=func_projection(fDat, CSP{i,1});
                ft=func_featureExtraction(tm, {'feature','logvar'});
                [cf_out(i)]=func_predict(ft, LDA{i,1});
            end
        else
            tm=func_projection(fDat, CSP);
            ft=func_featureExtraction(tm, {'feature','logvar'});
            [cf_out]=func_predict(ft, LDA)
        end
        %         cf_out
        if ~isempty(cf_out)
%             str=sprintf('%d    output: %f',a, cf_out);
        end
        if strcmp(opt.TCPIP, 'on')
                fwrite(d,cf_out,'double')
        end
        tic
    end
    %         a=a+1
    %     end
        pause(0.005)
end
end

