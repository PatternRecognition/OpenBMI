function [ epoch_biosignal ] = segmentationFatigue_RS( phy, mrk, interKSS )
% Segmentation of all types of data (EEG, PPG, KSS, RT) before calculation
% of correlation coefficient between differet tyeps of data
% All epoch is 1 seconds, power spectrum will be calculated based on EEG
% epoch, and epoch of other signals will be averaged

% Specific periods of measured data will be ruled out = 예외구간
% - 10 seconds of prior to KSS input (KSS score input periods)
% - Duration of deviation
% - 5 seconds after occurrence of collision
% - 5 seconds after course refresh at the end of drive course

bioData = phy.cnt;
sizeEpoch = 30;

stIdx = -255;
edIdx = 0;
kssIdx = 1; deviIdx = 1; refIdx = 2;
expFlag = 1;

epoch_biosignal.x = []; epoch_biosignal.misc = [];
epoch_biosignal.clab = phy.clab; epoIdx = 1;
epoch_biosignal.mClab{1} = 'KSS_BIO'; 

while 1
    %% 에포크 크기 설정
    if expFlag
        %         stIdx = edIdx + 1;
        stIdx = stIdx+256;
        edIdx = stIdx + (phy.fs * sizeEpoch) - 1;
    else
        expFlag = 1;
    end
    % 255 생성 시켜놓고 1초씩 겹치게
    
    % KSS 에포크 제거
    if ((mrk.pos(kssIdx) - stIdx < phy.fs) && (mrk.pos(kssIdx) - stIdx > 0)) ...
            || ((mrk.pos(kssIdx) - edIdx < phy.fs) && (mrk.pos(kssIdx) - edIdx > 0))
        expFlag = 0;    % - 10 seconds of prior to KSS input (KSS score input periods)
        % Start at after input of KSS score
        stIdx = mrk.pos(kssIdx) + 1;
        edIdx = stIdx + (phy.fs * sizeEpoch) - 1;
        
        if kssIdx < length(mrk.pos)
            kssIdx = kssIdx + 1;
        end
    end
    
    if expFlag
        epoch_biosignal.x(:, :, epoIdx) = [bioData(stIdx : edIdx, 1 : 4)];
        epoch_biosignal.misc(:, :, epoIdx) = [interKSS(stIdx : edIdx)];
        epoIdx = epoIdx + 1;
    end
    
    if edIdx + (phy.fs * sizeEpoch) > length(bioData);
        break;
    end
end
end

