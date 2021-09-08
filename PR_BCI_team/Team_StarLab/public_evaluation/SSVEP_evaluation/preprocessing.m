%%
% clear all;
% startup_bbci_toolbox
% 
% BTB.DataDir = 'E:\ear_data\isolated\data_1906';
% BTB.paradigm = 'SSVEP';
% 
% BTB.MatDir = [BTB.DataDir '\' BTB.paradigm ];
% 
% dire = [BTB.MatDir '\' '공인평가'];
% load([dire '\ambulatory_SSVEP_data_500'])

%% Make FFT features
for sub = 1:sum(~cellfun('isempty', cap_epo(:,1)))

%train = false;  % true: ispeed=1, false: ispeed=3
for ispeed = 1:sum(~cellfun('isempty', cap_epo(sub,:)))

epo_ = cap_epo{sub,ispeed};
% chan = {'P7', 'P3', 'P1','Pz','P2','P4','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};
chan = {'PO3','POz','PO4','O1','Oz','O2'};
% chan = {'Oz','POz'};

epo_ = proc_selectChannels(epo_, chan);
% for iseg = 1:length(segTime)
epo = proc_selectIval(epo_, [0 4000]);  % [0 4000]
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
% X_arr = X_arr(:,21:100,:);
len_freq = size(X_arr,2);
X_arr = X_arr(:,1:floor(len_freq/8),:);

%% get features
fv{sub,ispeed}.x = permute(X_arr,[2,3,4,1]);
fv{sub,ispeed}.y = nominal;

end
end
