function [ cca, avg ] = spectrumAnalysisSleep_R( epo, fRange, fs )
% Calculate the power spectrum density in specific frequency range
% 'epo' is vector of three dimension which have sample X channle X trials
% The return value is averaged values of power spectrum density in
% specific frequency range

subSize = 100;              % Size of sub-epochs
h = hamming(subSize);       % Creation of hamming window

for i = 1 : length(epo(1,1,:))
    clear subFFT subEpo ham_subEpo X;
    for j = 1 : 5 : 101     % Division of epoch into 21 sub-epochs
        subEpo = epo(j : j + subSize - 1, :, i);
        ham_subEpo = (subEpo .* repmat(h, 1, size(subEpo, 2)));
        
        %% Zero padding for epoch
        ham_subEpo = [zeros(floor(subSize / 2), size(ham_subEpo, 2)); ham_subEpo; ...
                      zeros(floor(subSize / 2), size(ham_subEpo, 2))];
        
        %% FFT for sub-epochs
        N = length(ham_subEpo);
        T = N / fs;                         % Get the frequency interval
        X = abs(fft(ham_subEpo)) / N * 2;   % Normalization of the data
        freq = [0 : N / 2 - 1] / T;         % Creation of the frequency range
        targetIdx = find(freq >= fRange(1) & freq <= fRange(2));
        
        subFFT(floor(j / 5) + 1, :, :) = X(targetIdx, :);
    end
    
    hammedFFT = squeeze(mean(subFFT, 1));
    cca(i, :) = mean(hammedFFT);

    avgFFT = squeeze(mean(hammedFFT, 1));
    powerDensity(i) = mean(avgFFT);
    avg = powerDensity';
end
end