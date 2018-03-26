function [ epoch ] = segmentationSleep( cnt, mrk, interKSS, interRT )
% Segmentation of all types of data (EEG, PPG, KSS, RT) before calculation
% of correlation coefficient between differet tyeps of data
% All epoch is 1 seconds, power spectrum will be calculated based on EEG
% epoch, and epoch of other signals will be averaged

% Specific periods of measured data will be ruled out = 예외구간
% - 10 seconds of prior to KSS input (KSS score input periods)
% - Duration of deviation
% - 5 seconds after occurrence of collision
% - 5 seconds after course refresh at the end of drive course

eegData = cnt.x(:, 1:32);
sizeEpoch = 1;

edIdx = 0;
kssIdx = 1; deviIdx = 1; refIdx = 2;
expFlag = 1;

epoch.x = []; epoch.misc = [];
epoch.clab(1 : 32) = cnt.clab(1 : 32); epoIdx = 1;
epoch.mClab{1} = 'KSS'; 

%% EEG frequency filtering
% [b, a] = butter(4, [0.5 50] / 100, 'bandpass');
% filtEEG = filter(b, a, eegData);
    filtEEG=cnt.x(:,1:32);

while 1
    
    %% 에포크 크기 설정
    if expFlag
        stIdx = edIdx + 1;
        edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
    else
        expFlag = 1;
    end
    
    % stldx와 edldx의 간격은 딱 200. 처음에 1에서 200까지
    
    % KSS 에포크 제거
    if ((mrk.kss.pos(kssIdx) - stIdx < 10 * cnt.fs) && (mrk.kss.pos(kssIdx) - stIdx > 0)) ...
            || ((mrk.kss.pos(kssIdx) - edIdx < 10 * cnt.fs) && (mrk.kss.pos(kssIdx) - edIdx > 0))
        expFlag = 0;    % - 10 seconds of prior to KSS input (KSS score input periods)
        % Start at after input of KSS score
        stIdx = mrk.kss.pos(kssIdx) + 1;
        edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
        
        if kssIdx < length(mrk.kss.pos)
            kssIdx = kssIdx + 1;
        end
        
        
        
        % RT 에포크 제거
    elseif ((stIdx < mrk.pos(deviIdx + 1) && stIdx > mrk.pos(deviIdx))) || ...
            ((edIdx < mrk.pos(deviIdx + 1) && edIdx > mrk.pos(deviIdx))) || ...
            (stIdx < mrk.pos(deviIdx) && edIdx > mrk.pos(deviIdx + 1))
        expFlag = 0;    % - Duration of deviation
        % Start at the end of deviation
        stIdx = mrk.pos(deviIdx + 1) + 1; edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
        
        if deviIdx < length(mrk.pos) - 1
            deviIdx = deviIdx + 2;
        end
        
        
        
        % 패러다임 시작, 끝, 충돌 등 에포크 제거
    elseif ((stIdx - mrk.misc.pos(refIdx) < 5 * cnt.fs) && (stIdx - mrk.misc.pos(refIdx) > 0)) ...
            || ((edIdx - mrk.misc.pos(refIdx) < 5 * cnt.fs) && (edIdx - mrk.misc.pos(refIdx) > 0))
        expFlag = 0;    % - 5 seconds after occurrence of collision
        % - 5 seconds after course refresh at the end of drive course
        % Start more than 5 seconds after collision of refresh
        stIdx = mrk.misc.pos(refIdx) + cnt.fs * 5 + 1; edIdx = stIdx + (cnt.fs * sizeEpoch) - 1;
        if refIdx < length(mrk.misc.pos) - 1
            refIdx = refIdx + 1;
        end
    end
    
    
    
    if expFlag
        % EEG(1 ~ 32), KSS, RT
        epoch.x(:, :, epoIdx) = [filtEEG(stIdx : edIdx, 1 : 32)];
        epoch.misc(:, :, epoIdx) = [interKSS(stIdx : edIdx)];
        epoIdx = epoIdx + 1;
    end
    
    if edIdx + (cnt.fs * sizeEpoch) > length(eegData);
        break;
    end
end
end

