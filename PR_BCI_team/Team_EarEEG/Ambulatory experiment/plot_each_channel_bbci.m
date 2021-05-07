function plot_each_channel_bbci(epo, segtime, varargin)
% plot each channel in segment time
% example: 
%           plot_each_channel_bbci(EPO, [0 30], 'nTrial',1 'data', EPO.x1 'scale', 200)
%
% input:    SMT
%           segtime:    a vector of two elements with 
%                       start point and end point
%           nTrial:     nth trial in interest
%           varargin:   
%               nTrial -    the turn of the class
%               data -      the signals in interest if you have other signals
%                           rather than SMT.x
%                           SMT.x, SMT.x1, SMT.x2, ...
%                           default: x
%               interval -  interval between plotting
% 

if isempty(epo)
    error('check your SMT data')
end
if ~isfield(epo, 'clab')
    warning('OpenBMI: Data must have a field named ''chan''')
    return
end
if ~isfield(epo, 'fs')
    warning('OpenBMI: Data must have a field named ''fs''')
    return
end


if isempty(segtime)
    segtime=[0 5];
end

opt = opt_proplistToStruct(varargin{:});

if isfield(opt,'nTrial')
    nTrial = opt.nTrial;
else
    nTrial =1;
end

if isfield(opt, 'channels')
    selected_ch = opt.channels;
    
    if ~prod(ismember(selected_ch, epo.clab))
        error('check selected channels')
    end
    
    ch_idx = ismember(epo.clab,selected_ch);
    epo.x = epo.x(:,ch_idx,:);
    epo.clab = epo.clab(ch_idx);
end

if isfield(opt,'data')
    data = opt.data;
    str_title = 'time domain plot';
else
    data = epo.x;
    str_title = 'time domain plot';
end

if isfield(opt,'title')
    str_title = opt.title;
end

if isfield(opt,'en_text')
    en_text = opt.en_text;
else
    en_text = true;
end

fs = epo.fs;

% time segment
t=segtime(1)+1/fs:1/fs:segtime(2);
size_data = size(data);
if size(size_data,2) == 3
    chVec = squeeze(data(int64(t*fs),:,nTrial));
elseif size(size_data,2) == 2
    if length(epo.clab) == size_data(2)
        chVec = squeeze(data(int64(t*fs),:));
    else
        chVec = squeeze(data(int64(t*fs),nTrial));
    end
end
    
% scale for plotting
windowSize = fs*0.01; b = (1/windowSize)*ones(1,windowSize); a = 1; 
t_dat_filt = filter(b,a,chVec);
plot_max = max(reshape(abs(t_dat_filt),1,[]));
if plot_max == 0
    plot_scale = 1;
    warning('plot scale is 0')
elseif plot_max < 10 && plot_max >= 4
    plot_scale = 10;
elseif  plot_max < 5 && plot_max >= 1
    plot_scale = 5;
elseif  plot_max < 2 && plot_max >=0.5
    plot_scale = 1;
elseif plot_max < 0.5
    plot_scale = plot_max*1.5;
else
    plot_scale = round((plot_max*1.5)/10)*10;
end

if isfield(opt,'scale')
    plot_scale = opt.scale;
end

% bias
bias = plot_scale * 2;
bias = cumsum(repmat(bias, 1, size(chVec,2)));

chVec = chVec + flip(bias);

% figure;
plot(t, chVec,'k')
% ylim([0-bias(1) bias(end)+bias(1)*2])
ylim([0 bias(end)+bias(1)])

xlim(segtime)
yticks(bias)
yticklabels(flip(epo.clab))
title(str_title);
ylabel('channels')
xlabel('time [s]')
str = sprintf('scale: %d', plot_scale);  % 
if en_text == true
annotation('textbox',[.01 .68 .3 .3],'String',str,'LineStyle','none');
end
grid on
grid minor
