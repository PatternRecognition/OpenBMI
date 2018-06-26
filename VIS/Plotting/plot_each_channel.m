function plot_each_channel(SMT, segtime, varargin)
% plot each channel in segment time
% example: 
%           plot_each_channel(SMT, [0 30], 2)
%
% input:    SMT
%           segtime:    a vector of two elements with 
%                       start point and end point
%           nTrial:     nth trial in interest
%           element:    the signals in interest if you have other signals
%                       rather than SMT.x
%                       SMT.x, SMT.x1, SMT.x2, ...
%                       default: x
%
% 

if isempty(SMT) || isempty(SMT.fs) || isempty(SMT.chan)
    error('check your SMT data')
end

if isempty(segtime)
    segtime=[0 5];
end
data = SMT.x;
nTrial =1;
if length(varargin) >= 2
    if ~isempty(varargin{1})
        nTrial = varargin{1};
    end
    data = varargin{2};
elseif length(varargin) == 1
    nTrial = varargin{1};
end

fs = SMT.fs;
% time segment
t=segtime(1)+1/fs:1/fs:segtime(2);
if size(size(data),2) == 3
    chVec = squeeze(data(int64(t*fs),nTrial,:));
elseif size(size(data),2) == 2
    chVec = squeeze(data(int64(t*fs),:));   
end
    
% scale for plotting
windowSize = fs*0.01; b = (1/windowSize)*ones(1,windowSize); a = 1; 
t_dat_filt = filter(b,a,chVec);
plot_scale = max(reshape(abs(t_dat_filt),1,[]));
plot_interval = plot_scale*3;

% bias
bias = plot_interval;
bias = cumsum(repmat(bias, 1, size(chVec,2)));

chVec = chVec + flip(bias);

figure;
plot(t, chVec,'b')
ylim([bias(1)-plot_interval bias(end)+plot_interval])
yticks(bias)
yticklabels(flip(SMT.chan))
title('time domain plot');
ylabel('channels')
xlabel('time [s]')
grid on
grid minor
