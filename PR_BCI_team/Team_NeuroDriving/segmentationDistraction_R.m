function [ epoch ] = segmentationDistraction( cnt, mrk )
% All epoch is 1 seconds, power spectrum will be calculated based on EEG
% epoch, and epoch of other signals will be averaged
i=1;
eegData = cnt.x(:, 1:64);
sizeEpoch = 1;

edIdx = 0;

deviIdx = 1; refIdx = 2;
expFlag = 1;

epoch.x = []; 
epoch.clab(1 : 64) = cnt.clab(1 : 64); epoIdx = 1;

%% EEG frequency filtering
% [b, a] = butter(4, [0.5 50] / 100, 'bandpass');
% filtEEG = filter(b, a, eegData);
filtEEG=cnt.x(:,1:64);

while 1
    
    %% 에포크 크기 설정
    if expFlag
        stIdx = edIdx + 1;
        edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
    else
        expFlag = 1;
    end
    
    if stIdx>45000
        
    
    
  
    % 주행 시작 지점에서 끝나는 지점까지만 지정
    if edIdx>mrk.pos(2*i)
        i=i+1;
        edIdx = mrk.pos(2*i-1)-1;
        stIdx = edIdx + 1;
        edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
    end
       
    if expFlag
        % EEG(1 ~ 32), KSS, RT
        epoch.x(:, :, epoIdx) = [filtEEG(stIdx : edIdx, 1 : 64)];
        epoIdx = epoIdx + 1;
    end
    end
    if i  == 119;
        break;
    end
        
end
end

