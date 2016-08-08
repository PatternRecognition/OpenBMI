clear all; close all; clc;

filelist= {'20160503_hblee_1'};


%% ------------------------------------------------------------------------------------------------------- %%
% Load CNN models for online classification of motor imagery
load(strcat('cnnModel', '_', filelist{1}));

%% ------------------------------------------------------------------------------------------------------- %%
% Online Initialization
params = struct;
state = bbci_acquire_bv('init', params);
EEG_data = [];
mnt = getElectrodePositions(state.clab);

epo.clab = state.clab;
epo.fs = state.fs;
epo.title = filelist{1};
epo.y = 0;

% Classification_Values = []; % for error correction
% cval = [];

order = 5;
Wps= [42 49]/epo.fs * 2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 40);
[filt.b, filt.a]= cheby2(n, 50, Ws);
% ----------------------------------------------------------------------
% UDP sender
ipA = 'RemoteIPAddress';
portA = 'RemoteIPPort';
ipB = '163.152.74.130';
portB = 5555;
udpbuff = 0;

udpB = udp(ipB, portB);
fopen(udpB);
udpB.Status
% ----------------------------------------------------------------------
% Get EEG data (vers. 2.0)
commandTime = [];
while 1
    data = bbci_acquire_bv(state);
    EEG_data = [EEG_data; data];
    si = size(EEG_data, 1);
    
    % Sampling Rate: 250Hz, get data every 3 sec
    if size(EEG_data,1) >= 751
        tic;
        epo.x = EEG_data(si - 751 + 1 : si, :);
        EEG_data = EEG_data(si - 751 + 1 : si, :);
        [filtEpo fBands fLevel] = prep_multibandFiltering(epo, order, fLimit, fWidth, fShift);

%         % EEG filtering
%         filtEpo = proc_filt(filtEpo, filt.b, filt.a);
        
        % Feature Extraction and Classification
        Classification_Result = onlineCNN_AGRCSP(filtEpo, rest_acsp_w, mi_acsp_w, rCspIdx, mCspIdx, ...
            rest_cnn, mi_cnn, fLevel, fBands, rest_pSize, mi_pSize);
        commandTime = [commandTime toc];
        
        
        % Error Correction
%         Classification_Values = [Classification_Values; Classification_Result];
%         sc = size(Classification_Values, 1);
        
%         if size(Classification_Values, 1) >= 11
%             cval = Classification_Values(sc - 11 + 1 : sc);
%             lefth = length(find(cval == 1));
%             righth = length(find(cval == 2));
%             foot = length(find(cval == 3));
%             rest = length(find(cval == 4));
%             
%             length_arr = [lefth righth foot rest];
%             [correctionMax correctionIdx] = max(length_arr);
%             Classification_Result = correctionIdx;
%         end
        
%         cval = [];
        
        % Classification
        switch Classification_Result
            case 1,
                disp('left');
                fwrite(udpB, uint8(11));     % SPEED Player1
            case 2,
                disp('right');
                fwrite(udpB, uint8(12));     % JUMP Player1
            case 3,
                disp('foot');
                fwrite(udpB, uint8(13));     % ROLL Player1
            otherwise,
                disp('rest');                % REST
                
        end
        udpB.ValuesSent
        
    end
    data = [];
end

fclose(udpB);
delete(udpB);
bbci_acquire_bv('close');