function output = vis_plotCont(SMT, varargin)
% Description:
%   Draw  scalp topographies for all selected intervals,separately for each each class.
%   Scalp topographies of each classe are plotted in one row, and shared the same color map
%   scaling in each classes.
%
% Example Code:
%    visual_scalpPlot(SMT,CNT, {'Ival' , [start interval : time increase parameter: end intercal]});
%
% Input:
%   visual_scalpPlot(SMT,CNT, <OPT>);
%   SMT: Data structrue (ex) Epoched data structure
%   CNT: Continuous data structure
%
% Option:
%      .Ival - Selecting the interested time interval depending on time increase parameter
%                 (e.g. {'Ival' [ [-2000 : 1000: 4000]])
%
% Return:
%   Scalp topographies
%
% See also:
%    opt_getMontage, opt_cellToStruct
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr
% Hong Kyung, Kim
% hk_kim@korea.ac.kr

% -----------------------------------------------------
% FileExchange function subtightplot by F. G. Nievinski
subplot = @(m,n,p) subtightplot(m,n,p,[0.045 0], 0.05, [0.06, 0.048]);
output_str = {'';'';'Finished'};

%%Options
opt = opt_cellToStruct(varargin{:});
if isfield(opt, 'Interval') 
    interval = opt.Interval; 
else    
    opt.Interval = [SMT.ival(1) SMT.ival(end)];
    interval = opt.Interval; 
end
if isfield(opt, 'Range') 
    plot_range = opt.Range; 
else
    opt.Range = 'sym'; 
    plot_range = opt.Range;
end
if isfield(opt, 'Channels') 
    chan = opt.Channels; 
else
    opt.Channels = {SMT.chan{1:2}}; 
    chan = opt.Channels;
end
if isfield(opt, 'Class') 
    class = opt.Class; 
else
    opt.Class = {SMT.class{1:2,2}}; 
    class = opt.Class; 
end
if isfield(opt, 'Patch') 
    Patch = opt.Patch; 
else
    opt.Patch = 'on';
    Patch = opt.Patch; 
end
if isfield(opt, 'TimePlot') 
    TimePlot = opt.TimePlot; 
else
    opt.TimePlot = 'off';
    TimePlot = opt.TimePlot; 
end
if isfield(opt, 'ErspPlot') 
    ErspPlot = opt.ErspPlot;
else
    opt.ErspPlot = 'off';
    ErspPlot = opt.ErspPlot;
end
if isfield(opt, 'ErdPlot') 
    ErdPlot = opt.ErdPlot; 
else
    opt.ErdPlot = 'off'; 
    ErdPlot = opt.ErdPlot; 
end
if isfield(opt, 'TopoPlot') 
    TopoPlot = opt.TopoPlot; 
else
    opt.TopoPlot = 'on'; 
    TopoPlot = opt.TopoPlot; 
end
if isfield(opt, 'FFTPlot') 
    FFTPlot = opt.FFTPlot; 
else
    opt.FFTPlot = 'off'; 
    FFTPlot = opt.FFTPlot; 
end
if isfield(opt, 'Baseline') 
    baseline = opt.Baseline; 
else
     opt.Baseline = [0 0]; 
    baseline = opt.Baseline; 
end
if ~isfield(opt, 'SelectTime')
    opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; 
end
if isfield(opt, 'Align') 
    align = opt.Align; 
else
    opt.Align = 'vert'; 
    align = opt.Align; 
end

%% Figure Settings
fig = figure('Color', 'w');
set(fig, 'ToolBar', 'none');

monitor_screensize = get(0, 'screensize');


set(gcf,'Position',[monitor_screensize(3)/4,  monitor_screensize(4)/4,...
    monitor_screensize(3)/2, monitor_screensize(4)/2]);

oldUnit = get(gcf,'units');
set(gcf,'units','normalized');

%% Preparing subplots
num_Plot = sum(ismember({TimePlot, ErspPlot, ErdPlot}, 'on')) + ismember({FFTPlot}, 'on') * length(class);

global plotOptions;
if isequal(align, 'vert') || xor(isequal(TopoPlot, 'on'), num_Plot)
    plotOptions.plot_ratio = 1;
    if size(interval, 1) * size(class, 1) > 5
        plotOptions.sub_row = length(chan) * num_Plot + size(class,1) * isequal(TopoPlot, 'on');
        plotOptions.sub_col = size(interval,1);
    else
        plotOptions.sub_row = length(chan) * num_Plot + 1;
        plotOptions.sub_col = size(interval, 1) * size(class, 1);
    end
    plotOptions.plot_sub_temp = [1 plotOptions.sub_col];
    plotOptions.topo_sub_temp = 1;
    plotOptions.plot_sub_hop = plotOptions.sub_col;
    plotOptions.topo_sub_hop = 1;
elseif isequal(align, 'horz')
    plotOptions.plot_ratio = 3;
    plotOptions.sub_row = num_Plot * length(chan) * size(class,1);
    plotOptions.sub_col = size(interval,1) * plot_ratio * size(class,1);
end

%% time-domain plot
if isequal(TimePlot, 'on')
    erders_plt = vis_timeAveragePlot(SMT, opt);
    grp_ylabel(time_plt, 'Time-Domain');
end
%% ERDERS plot
if isequal(ErdPlot, 'on')
    opt.Envelope = true;
    erders_plt = vis_timeAveragePlot(SMT, opt);
    grp_ylabel(erders_plt, 'ERD/ERS');
end
%% ERSP plot
if isequal(ErspPlot, 'on')
end
%% FFT Plot
if isequal(FFTPlot , 'on')
    fft_plt = vis_freqFFTPlot(SMT, opt);
    grp_ylabel(fft_plt, 'FFT');
end
%% Topo plot
if isequal(TopoPlot, 'on')
    topo_plt = vis_topoPlot(SMT, opt);
    grp_ylabel(topo_plt, 'Topography');
end
output = output_str;
end

function grp_ylabel(plots, title)
pos = get([plots{:}], 'Position');
if iscell(pos), pos = cell2mat(pos); end
axes('Position',[min(pos(:,1))*0.8, min(pos(:,2)), min(pos(:,1))*0.2,...
    abs(max(pos(:,2))-min(pos(:,2)))+max(pos(:,4))], 'Visible', 'off');
set(get(gca,'Ylabel'), 'Visible','on', 'String', title, ...
    'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 12);
end


function [X,freq]=positiveFFT(x,Fs)
N=length(x);    % get the number of points
k=0:N-1;        % create a vector from 0 to N-1
T=N/Fs;         % get the frequency interval
freq=k/T;       % create the frequency range
X=fft(x)/N*2;   % normalize the data
cutOff = ceil(N/2);

% take only the first half of the spectrum
X = X(1:cutOff);
freq = freq(1:cutOff);
end

function grp_plots = vis_timeAveragePlot(SMT, varargin)
% Description:
%
% Input:
%       timeSMT: epoched EEG data
% Options:
%       'Channels',
%       'Class',
%       'Rnage',
%       'Baseline',
%       'Envelope',
% Ouput:
%       grp_plots : cell array of axes
% Example:
%       Options = {'Channels', {'Cz', 'Oz'};
%                   'Class', {'target'; 'non_target'};
%                   'Range', [100 200];
%                   'Baseline', [-200 1000];
%                   'Envelope', false};
%       axis = vis_timeAveragePlot(SMT, Options);
%
% Created by Hong Kyung, Kim
% hk_kim@korea.ac.kr

switch nargin
    case 1
        opt = [];
    case 2
        opt = varargin{:};
        if ~isstruct(opt)
            opt = opt_cellToStruct(opt);
        end
end

faceColor = [{[0.8 0.8 0.8]};{[0.8 0.8 0.8]}];
%% Options
if ~isfield(opt, 'Channels') opt.Channels = {SMT.chan{1:5}}; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1:2,2}}; end
if ~isfield(opt, 'Baseline') opt.Baseline = [SMT.ival(1) SMT.ival(1)]; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'Interval') opt.Interval = [opt.SelectTime(1) opt.SelectTime(end)]; end
if ~isfield(opt, 'Patch') opt.Patch = 'off'; end

global plotOptions;
if ~isstruct(plotOptions)
    plotOptions.plot_ratio = 1;
    plotOptions.sub_row = length(opt.Channels);
    plotOptions.sub_col = 1;
    plotOptions.plot_sub_temp = [1 plotOptions.sub_col];
    plotOptions.plot_sub_hop = plotOptions.sub_col;
end

grp_plots = cell(1, length(opt.Channels));
SMT = prep_selectChannels(SMT, {'Name',  opt.Channels});
SMT = prep_selectClass(SMT,{'Class', opt.Class});
if isfield(opt, 'Envelope') && opt.Envelope
    SMT = prep_envelope(SMT);
end
SMT = prep_baseline(SMT, {'Time', opt.Baseline});
SMT = prep_selectTime(SMT, {'Time', opt.SelectTime});
SMT = prep_average(SMT);

if isfield(opt, 'TimeRange') && ~isempty(opt.TimeRange)
    time_range = opt.TimeRange;
else
    time_range = [floor(min(reshape(SMT.x, [], 1))),...
        ceil(max(reshape(SMT.x, [], 1)))]*1.2;
end

for ch_num = 1:length(opt.Channels)
    grp_plots{ch_num} = subplot(plotOptions.sub_row, plotOptions.sub_col, plotOptions.plot_sub_temp);
    
    plot(SMT.ival, SMT.x(:,:,ch_num),'LineWidth',2); hold on;
    legend(opt.Class, 'Interpreter', 'none', 'AutoUpdate', 'off');
    %         set({'color'}, co(1:size(avgSMT.class, 1)));
    
    grid on;
    ylim(time_range);
    
    if isequal(opt.Patch, 'on')
        % baselin patch
        base_patch = min(abs(time_range))*0.05;
        patch('XData', [baseline(1)  baseline(2) baseline(2) baseline(1)], ...
            'YData', [-base_patch -base_patch base_patch base_patch],...
            'FaceColor', 'k',...
            'FaceAlpha', 0.7, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        % ival patch
        for ival = 1:size(opt.Interval,1)
            patch('XData', [opt.Interval(ival,1) opt.Interval(ival,2) opt.Interval(ival,2) opt.Interval(ival,1)],...
                'YData', [time_range(1) time_range(1) time_range(2) time_range(2)],...
                'FaceColor', faceColor{mod(ival,2)+1},...
                'FaceAlpha', 0.3, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        end
        
        tmp = get(grp_plots{ch_num}, 'Children');
        set(grp_plots{ch_num}, 'Children', flip(tmp));
    end
    ylabel(opt.Channels{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
    hold off;
    plotOptions.plot_sub_temp = plotOptions.plot_sub_temp + plotOptions.sub_col;
end
end

function grp_plots = vis_freqFFTPlot(SMT, varargin)

switch nargin
    case 1
        opt = [];
    case 2
        opt = varargin{:};
        if ~isstruct(opt)
            opt = opt_cellToStruct(opt);
        end
end

%% Options
if ~isfield(opt, 'Channels') opt.Channels = {SMT.chan{1}}; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1:2,2}}; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end

global plotOptions;
if ~isstruct(plotOptions)
    plotOptions.plot_ratio = 1;
    plotOptions.sub_row = length(opt.Channels);
    plotOptions.sub_col = 1;
    plotOptions.plot_sub_temp = [1 plotOptions.sub_col];
    plotOptions.plot_sub_hop = plotOptions.sub_col;
end

grp_plots = cell(1, length(opt.Channels) * length(opt.Class));
SMT = prep_selectChannels(SMT, {'Name', opt.Channels});
SMT = prep_selectClass(SMT, {'class', opt.Class});
SMT = prep_selectTime(SMT, {'Time', opt.SelectTime});
SMT = prep_average(SMT);

co = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250;...
    0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330;...
    0.6350 0.0780 0.1840];


for ch_num = 1:length(opt.Channels)
    for cl = 1:length(opt.Class)
        grp_plots{cl} = subplot(plotOptions.sub_row, plotOptions.sub_col, plotOptions.plot_sub_temp);
        
        [YfreqDomain,frequencyRange] = positiveFFT(SMT.x(:,cl, ch_num),SMT.fs);
        
        plot(frequencyRange,abs(YfreqDomain), 'Color', co(cl, :)); hold on;
        
        legend(SMT.class(cl,2), 'Interpreter', 'none', 'AutoUpdate', 'off');
        
        grid on;
        
        ylabel(SMT.chan{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
        
        plotOptions.plot_sub_temp = plotOptions.plot_sub_temp + plotOptions.sub_col;
    end
end


function [X,freq]=positiveFFT(x,Fs)
N=length(x);    % get the number of points
k=0:N-1;        % create a vector from 0 to N-1
T=N/Fs;         % get the frequency interval
freq=k/T;       % create the frequency range
X=fft(x)/N*2;   % normalize the data
cutOff = ceil(N/2);

% take only the first half of the spectrum
X = X(1:cutOff);
freq = freq(1:cutOff);
end
end

function grp_plots = vis_topoPlot(SMT, varargin)
% Description:
%
%
%
%
%
%
%
PREVENT_OVERFLOW = 10000;
%% Options

switch nargin
    case 1
        opt = [];
    case 2
        opt = varargin{:};
        if ~isstruct(opt)
            opt = opt_cellToStruct(opt);
        end
end
if ~isstruct(opt)
    opt = opt_cellToStruct(varargin{:});
end

%% Creating Montage 
MNT = opt_getMontage(SMT);
output_str = [];

if ~isequal(MNT.chan, SMT.chan)
    if length(SMT.chan) > length(MNT.chan)
        tmp = SMT.chan(~ismember(SMT.chan, MNT.chan));
        output_str = tmp{1};
        for i = 2:length(tmp)
            output_str = [output_str sprintf(', %s',tmp{i})];
        end
        output_str = {''; [output_str, ' are missed']};
    end
    tmp = zeros(1,length(MNT.x));
    for i = 1:length(MNT.chan)
        tmp(i) = find(ismember(SMT.chan, MNT.chan{i}));
    end
    SMT.x = SMT.x(:,:,tmp);
    SMT.chan = SMT.chan(tmp);
    clear tmp;
end
    
if ~isfield(opt, 'Colormap') opt.Colormap = 'parula'; end
if ~isfield(opt, 'Quality') opt.Quality = 'high'; end
if ~isfield(opt, 'Class') opt.Class = {SMT.class{1:2,2}}; end
if ~isfield(opt, 'Baseline') opt.Baseline = [SMT.ival(1) SMT.ival(1)]; end
if ~isfield(opt, 'SelectTime') opt.SelectTime = [SMT.ival(1) SMT.ival(end)]; end
if ~isfield(opt, 'Interval') opt.Interval = [opt.SelectTime(1) opt.SelectTime(end)]; end
if ~isfield(opt, 'Channels') opt.Channels = {SMT.chan{1:5}}; end
if ~isfield(opt, 'Range') opt.Range = 'sym'; end

global plotOptions;
if ~isstruct(plotOptions)
    plotOptions.plot_ratio = 1;
    plotOptions.sub_row = length(opt.Channels);
    plotOptions.sub_col = 1;
    plotOptions.plot_sub_temp = 1;
    plotOptions.plot_sub_hop = plotOptions.sub_col;
end

if ~isequal(plotOptions.plot_sub_temp(1), 1)
    plotOptions.topo_sub_temp = plotOptions.plot_sub_temp(1);
end

% scalp_plot quality
switch lower(opt.Quality)
    case 'high'
        resol = 256;
    case 'medium'
        resol = 144;
    case 'low'
        resol = 32;
    otherwise
        resol = 256;
end
% Scalp colormap;
colormap(opt.Colormap);

SMT = prep_selectClass(SMT ,{'class',opt.Class});
if size(SMT.x, 2) > PREVENT_OVERFLOW
    divis = divisors(size(SMT.x, 2));
    divis = divis(divis < PREVENT_OVERFLOW);
    divis = divis(end);
    SMT_size = size(SMT.x, 2);
    for i = 1:divis:SMT_size
        tmpSMT = prep_selectTrials(SMT, {'Index', i:i+divis-1});
%         tmpSMT = prep_selectClass(tmpSMT ,{'class',opt.Class});
        if opt.Envelope
            tmpSMT = prep_envelope(tmpSMT);
        end
        tmpSMT = prep_baseline(tmpSMT, {'Time', opt.Baseline});
        SMT.x(:,i:i+divis-1,:) = tmpSMT.x;
    end
    clear tmpSMT;
else
    SMT = prep_selectClass(SMT,{'class',opt.Class});
     if isfield(opt, 'Envelope') && opt.Envelope
        SMT = prep_envelope(SMT);
    end
    SMT = prep_baseline(SMT, {'Time', opt.Baseline});
end
SMT = prep_selectTime(SMT, {'Time', opt.SelectTime});
SMT = prep_average(SMT);

for i = 1: size(opt.Class, 1)
    ivalSegment = size(opt.Interval,1);
    minmax = [];
    for seg = 1: ivalSegment
        SMTintervalstart = find(SMT.ival == opt.Interval(seg,1));
        SMTintervalEnd = find(SMT.ival == opt.Interval(seg,2));
        ivalSMT = squeeze(SMT.x(SMTintervalstart:SMTintervalEnd,i,:));
        w{i, seg} = mean(ivalSMT,1);
        minmax = [minmax; min(w{i,seg}(:)), max(w{i,seg}(:))];
    end
    %% range_options
    if ~isfloat(opt.Range)
        switch opt.Range
            case 'sym'
                p_range = [-max(abs(max(minmax(:,2)))), max(abs(max(minmax(:,2))))];
            case '0tomax'
                p_range = [0.0001*diff([min(minmax(:,1)) max(minmax(:,2))]), max(minmax(:,2))];
            case 'minto0'
                p_range = [min(minmax(:,1)), 0.0001*diff([min(minmax(:,1)) max(minmax(:,2))])];
            case 'mintomax'
                p_range = [min(minmax(:,1)) max(minmax(:,2))];
            case 'mean'
                p_range = [mean(minmax(:,1)) mean(minmax(:,2))];
        end
    else
        p_range = plot_range;
    end
    if diff(p_range)==0
        p_range(2)= p_range(2)+eps;
    end
    %% Draw
    for seg = 1: size(opt.Interval, 1)
        grp_plots{i,seg} = subplot(plotOptions.sub_row, plotOptions.sub_col, plotOptions.topo_sub_temp);
        plot_scalp(grp_plots{i,seg}, w{i, seg}, MNT, p_range, resol);
        xlabel(sprintf('[%d - %d] ms',opt.Interval(seg,:)), 'FontWeight', 'normal');
        plotOptions.topo_sub_temp = plotOptions.topo_sub_temp + plotOptions.topo_sub_hop;
    end
    % Keeping scalp size;
    last_position = get(grp_plots{i,seg}, 'Position');
    colorbar('vert');
    tmp = get(grp_plots{i,seg}, 'Position');
    tmp(3:4) = last_position(3:4);
    set(gca,'Position',tmp);
    subplot(plotOptions.sub_row, plotOptions.sub_col, plotOptions.topo_sub_temp - size(opt.Interval,1) * plotOptions.topo_sub_hop);
    set(get(gca,'Ylabel'), 'Visible','on', 'String', {opt.Class{i};''}, ...
        'Interpreter', 'none', 'FontWeight', 'normal', 'FontSize', 12);
end
end


function grpSubPlots = vis_subplotTemplate(subrow, subcol, columns)
if nargin == 2 || columns < 1
    columns = 1;
end
num_plot = 10;
time_range = 5;
class = 7;
sel_chan = 5;

plot_ratio = 2;
sub_row = num_plot * sel_chan * class;
sub_col = time_range * plot_ratio * class;

plot_sub_temp = [];
for i = 1:sub_row / (sel_chan * num_plot)
    plot_sub_temp = horzcat(plot_sub_temp, [1 sub_col/plot_ratio] + ((i-1) * sub_col));
end

topo_sub_temp = [];
for i = 1:sub_row / class
    topo_sub_temp = horzcat(topo_sub_temp, [sub_col/plot_ratio+1 sub_col/plot_ratio + sub_col/(plot_ratio*time_range)*(plot_ratio - 1)] + ((i-1) * sub_col));
end

sub_plot_hop = sub_row / (sel_chan * num_plot) * sub_col;

for i = 1:num_plot * sel_chan
    subplot(sub_row, sub_col, plot_sub_temp + (i-1) * sub_plot_hop);
end

sub_topo_hop = [sub_col/(plot_ratio*time_range) * (plot_ratio - 1) sub_row / class * sub_col];

for i = 1:class
    for j = 1:time_range
        subplot(sub_row, sub_col, topo_sub_temp + sub_topo_hop(1) * (j-1));
    end
    topo_sub_temp = topo_sub_temp + sub_topo_hop(2);
end
end

%% FileExchange function subtightplot by F. G. Nievinski
function h=subtightplot(m,n,p,gap,marg_h,marg_w,varargin)
%function h=subtightplot(m,n,p,gap,marg_h,marg_w,varargin)
%
% Functional purpose: A wrapper function for Matlab function subplot. Adds the ability to define the gap between
% neighbouring subplots. Unfotrtunately Matlab subplot function lacks this functionality, and the gap between
% subplots can reach 40% of figure area, which is pretty lavish.
%
% Input arguments (defaults exist):
%   gap- two elements vector [vertical,horizontal] defining the gap between neighbouring axes. Default value
%            is 0.01. Note this vale will cause titles legends and labels to collide with the subplots, while presenting
%            relatively large axis.
%   marg_h  margins in height in normalized units (0...1)
%            or [lower uppper] for different lower and upper margins
%   marg_w  margins in width in normalized units (0...1)
%            or [left right] for different left and right margins
%
% Output arguments: same as subplot- none, or axes handle according to function call.
%
% Issues & Comments: Note that if additional elements are used in order to be passed to subplot, gap parameter must
%       be defined. For default gap value use empty element- [].
%
% Usage example: h=subtightplot((2,3,1:2,[0.5,0.2])

if (nargin<4) || isempty(gap),    gap=0.01;  end
if (nargin<5) || isempty(marg_h),  marg_h=0.05;  end
if (nargin<5) || isempty(marg_w),  marg_w=marg_h;  end
if isscalar(gap),   gap(2)=gap;  end
if isscalar(marg_h),  marg_h(2)=marg_h;  end
if isscalar(marg_w),  marg_w(2)=marg_w;  end
gap_vert   = gap(1);
gap_horz   = gap(2);
marg_lower = marg_h(1);
marg_upper = marg_h(2);
marg_left  = marg_w(1);
marg_right = marg_w(2);

%note n and m are switched as Matlab indexing is column-wise, while subplot indexing is row-wise :(
[subplot_col,subplot_row]=ind2sub([n,m],p);

% note subplot suppors vector p inputs- so a merged subplot of higher dimentions will be created
subplot_cols=1+max(subplot_col)-min(subplot_col); % number of column elements in merged subplot
subplot_rows=1+max(subplot_row)-min(subplot_row); % number of row elements in merged subplot

% single subplot dimensions:
%height=(1-(m+1)*gap_vert)/m;
%axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh;
height=(1-(marg_lower+marg_upper)-(m-1)*gap_vert)/m;
%width =(1-(n+1)*gap_horz)/n;
%axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
width =(1-(marg_left+marg_right)-(n-1)*gap_horz)/n;

% merged subplot dimensions:
merged_height=subplot_rows*( height+gap_vert )- gap_vert;
merged_width= subplot_cols*( width +gap_horz )- gap_horz;

% merged subplot position:
merged_bottom=(m-max(subplot_row))*(height+gap_vert) +marg_lower;
merged_left=(min(subplot_col)-1)*(width+gap_horz) +marg_left;
pos_vec=[merged_left merged_bottom merged_width merged_height];

% h_subplot=subplot(m,n,p,varargin{:},'Position',pos_vec);
% Above line doesn't work as subplot tends to ignore 'position' when same mnp is utilized
h=subplot('Position',pos_vec,varargin{:});

if (nargout < 1),  clear h;  end

end