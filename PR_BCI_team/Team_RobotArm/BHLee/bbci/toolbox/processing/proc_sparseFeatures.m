function [ fv, opt ] = proc_sparseFeatures( epo, ivals, deltas)
%PROC_SPARSEFEATURES 
% extracts sparse samples from epo: for any ival(i), take a sample every
% deltas(i) ms.
%     IVALS   Intervals sampled from (DEFAULT: [100, 350; 350 700])
%     DELTAS  time (in ms) between two sample points per ival. Has to be a
%     multiple of 1000/epo.fs  (DEFAULT: [20; 50]) 
%
% Johannes 06/2011
epo_in = epo;
indices = [];
if nargin < 3
    ivals = [100, 350; 350 700];
    deltas = [20; 50];
end

baseDelta = 1000/epo.fs;
if sum(mod(deltas, baseDelta) == 0) ~= length(deltas)
    error('deltas have to be a multiple of %02d ms (=1000/epo.fs)', 1000/epo.fs)
end
if ~unique(1000./diff(epo.t) == epo.fs)
    error('time (.t) dont correspond to fs')
end

st = deltas / baseDelta; % sampling steps. one for each ival
maxf = 1000/min(deltas);

%% low-pass filtering
    Wps= [round(.4*maxf) round(.48 * maxf)]/ epo.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
    [filt.b, filt.a]= cheby2(n, 50, Ws);
    epo = proc_filtfilt(epo, filt.b, filt.a);
    
%% select features
for k=1:size(ivals, 1)
    data_idx = find(epo.t >= ivals(k,1) & ...
        epo.t <= ivals(k,2)); % find the interval indices
    data_idx = data_idx(1:st(k):end); % sample the interval
    indices = [indices, data_idx];
end
indices = unique(indices); %remove doubles that possibly happened
epo.x = epo.x(indices, :,:);
epo.t = epo.t(indices);

% flaten and normalize
opt = [];
[fv, opt.meanOpt] = proc_subtractMean(proc_flaten(epo));
[fv, opt.normOpt] = proc_normalize(fv);
