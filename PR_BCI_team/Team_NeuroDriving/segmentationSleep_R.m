function [ epoch ] = segmentationSleep( cnt, mrk, interKSS )
% Segmentation of all types of data (EEG, PPG, KSS, RT) before calculation
% of correlation coefficient between differet tyeps of data
% All epoch is 1 seconds, power spectrum will be calculated based on EEG
% epoch, and epoch of other signals will be averaged

% Specific periods of measured data will be ruled out = 예외구간
% - 10 seconds of prior to KSS input (KSS score input periods)
% - Duration of deviation
% - 5 seconds after occurrence of collision
% - 5 seconds after course refresh at the end of drive course

eegData = cnt.x(:, 1:64);
sizeEpoch = 1;

edIdx = 0;
kssIdx = 1; deviIdx = 1; refIdx = 2;
expFlag = 1;

epoch.x = []; epoch.misc = [];
epoch.clab(1 : 64) = cnt.clab(1 : 64); epoIdx = 1;
epoch.mClab{1} = 'KSS';

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
    
    % KSS 에포크 제거
    if ((mrk.pos(kssIdx) - stIdx < 10 * cnt.fs) && (mrk.pos(kssIdx) - stIdx > 0)) ...
            || ((mrk.pos(kssIdx) - edIdx < 10 * cnt.fs) && (mrk.pos(kssIdx) - edIdx > 0))
        expFlag = 0;    % - 10 seconds of prior to KSS input (KSS score input periods)
        % Start at after input of KSS score
        stIdx = mrk.pos(kssIdx) + 1;
        edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
        
        if kssIdx < length(mrk.pos)
            kssIdx = kssIdx + 1;
        end
    end
        
    if expFlag
        % EEG(1 ~ 32), KSS, RT
        epoch.x(:, :, epoIdx) = [filtEEG(stIdx : edIdx, 1 : 64)];
        epoch.misc(:, :, epoIdx) = [interKSS(stIdx : edIdx)];
        epoIdx = epoIdx + 1;
    end
    
    if edIdx + (cnt.fs * sizeEpoch) > length(eegData);
        break;
    end
end
end

