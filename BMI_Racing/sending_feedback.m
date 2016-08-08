
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];

d = tcpip('localhost', 3000, 'NetworkRole', 'Client');
set(d, 'OutputBufferSize', 1024); % Set size of receiving buffer, if needed.

%Trying to open a connection to the server.
while(1)
    try
        fopen(d);
        break;
    catch
        fprintf('%s \n','Cant find Server');
    end
end
connectionSend = d;


%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

%% if you can redefine the marker information after Load_EEG function
%% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});

%% PRE-PROCESSING MODULE
filter=[7 13];
CNT=prep_filter(CNT, {'frequency', filter});
SMT=prep_segmentation(CNT, {'interval', [1010 3000]});
SMT=prep_selectChannels(SMT, {'Index',[1:20]});


%% SPATIAL-FREQUENCY OPTIMIZATION MODULE
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
FT=func_featureExtraction(SMT, {'feature','logvar'});

%% CLASSIFIER MODULE
[CF_PARAM]=func_train(FT,{'classifier','LDA'});

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);

buffer_size=state.fs*5;
feedback_t=100/1000; % feedback frequency
Dat=zeros(buffer_size, size(state.chan_sel,2));
[nDat nTR nCH]=size(SMT.x);

escapeKey = KbName('esc');
waitKey=KbName('s');

tic
while true
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            break;
        elseif keyCode(waitKey)
            break;
        end
    end
    
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    Dat=[Dat; data];
    if length(Dat)>buffer_size % prevent overflow
        Dat=Dat(end-buffer_size+1:end,:);
    end
    if toc>feedback_t
        fDat=prep_filter(Dat, {'frequency', filter;'fs',state.fs });
        fDat=func_projection(fDat, CSP_W);
        feature=func_featureExtraction(fDat, {'feature','logvar'});
        [cf_out]=func_predict(feature, CF_PARAM);
        str=sprintf('output: %f',cf_out)
        fwrite(d,cf_out);
        tic
    end
end

