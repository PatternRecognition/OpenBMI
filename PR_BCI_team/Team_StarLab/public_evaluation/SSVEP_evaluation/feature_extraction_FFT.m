function [epo_X, epo_Y] = feature_extraction_FFT(epo)

chan = {'PO3','POz','PO4','O1','Oz','O2'};

epo = proc_selectChannels(epo, chan);
epo = proc_selectIval(epo, [0 4000]);  % [0 4000]
dataset = permute(epo.x, [3,1,2]);

[tr, dp, ch] = size(dataset); % tr: trial, dp: time, ch: channel

nominal = [];
for i=1:size(epo.y,2)
    nominal(i) = find(epo.y(:,i),1)-1;
end


%% Fast Fourier Transform (FFT)
X_arr=[]; % make empty array
for k=1:tr % trials
    x=squeeze(dataset(k, :,:)); % data
    N=length(x);    % get the number of points
    kv=0:N-1;        % create a vector from 0 to N-1
    T=N/epo.fs;         % get the frequency interval
    freq=kv/T;       % create the frequency range
    X=fft(x)/N*2;   % normalize the data
    cutOff = ceil(N/2); % get the only positive frequency
    
    % take only the first half of the spectrum
    X=abs(X(1:cutOff,:)); % absolute values to cut off
    freq = freq(1:cutOff); % frequency to cut off
    XX = permute(X,[3 1 2]);
    X_arr=[X_arr; XX]; % save in array
end

%% frequency band
% f_gt = [11 7 5];  % 5.45, 8.75, 12
f_last = find( freq > 30, 1); % freq < 30Hz
X_arr = X_arr(:,1:f_last,:); %

%% get features
epo_X = permute(X_arr,[2,3,4,1]);
epo_Y = nominal;

