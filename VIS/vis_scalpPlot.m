function output = vis_scalpPlot(SMT, varargin)
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

opt = opt_cellToStruct(varargin{:});
if isfield(opt, 'Interval') interval = opt.Interval; end
if isfield(opt, 'Range') plot_range = opt.Range; else plot_range = 'sym'; end
if isfield(opt, 'Quality') quality = opt.Quality; else quality = 'high'; end
if isfield(opt, 'Channels') chan = opt.Channels; else chan = {SMT.chan{1:2}}; end
if isfield(opt, 'Class') class = opt.Class; else class = {SMT.class{1:2,2}}; end
if isfield(opt, 'Patch') Patch = opt.Patch; end
if isfield(opt, 'TimePlot') TimePlot = opt.TimePlot; else TimePlot = 'off'; end
if isfield(opt, 'ErspPlot') ErspPlot = opt.ErspPlot; else ErspPlot = 'off'; end
if isfield(opt, 'ErdPlot') ErdPlot = opt.ErdPlot; else ErdPlot = 'off'; end
if isfield(opt, 'TopoPlot') TopoPlot = opt.TopoPlot; else TopoPlot = 'on'; end
if isfield(opt, 'FFTPlot') FFTPlot = opt.FFTPlot; else FFTPlot = 'off'; end 
if isfield(opt, 'Baseline') baseline = opt.Baseline; else baseline = [0 0]; end
if isfield(opt, 'SelectTime')seltime = opt.SelectTime; else seltime = [SMT.ival(1) SMT.ival(end)]; end
if isfield(opt, 'Colormap') cm = opt.Colormap; else cm = 'parula'; end
if isfield(opt, 'Align') align = opt.Align; else align = 'vert'; end
if isfield(opt, 'Colororder') co = opt.Colororder; else co = {[ 0 0.4470 0.7410];...
        [0.8500 0.3250 0.0980]; [0.9290 0.6940 0.1250]; [0.4940 0.1840 0.5560]; [0.4660 0.6740 0.1880];};end

fig = figure('Color', 'w');
set(fig, 'ToolBar', 'none');

monitor_screensize = get(0, 'screensize');


set(gcf,'Position',[monitor_screensize(3)/4,  monitor_screensize(4)/4,...
    monitor_screensize(3)/2, monitor_screensize(4)/2]);
MNT = opt_getMontage(SMT);

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

if isempty(interval)
    interval = [SMT.ival(1) SMT.ival(end)];
end

oldUnit = get(gcf,'units');
set(gcf,'units','normalized');

colormap(cm);
faceColor = [{[0.8 0.8 0.8]};{[0.8 0.8 0.8]}];

num_Plot = sum(ismember({TimePlot, ErspPlot, ErdPlot}, 'on')) + ismember({FFTPlot}, 'on') * length(class);

if isequal(align, 'vert') || xor(isequal(TopoPlot, 'on'), num_Plot)
    plot_ratio = 1;
    if size(interval, 1) * size(class, 1) > 5
        sub_row = length(chan) * num_Plot + size(class,1) * isequal(TopoPlot, 'on');
        sub_col = size(interval,1);
    else
        sub_row = length(chan) * num_Plot + 1;
        sub_col = size(interval, 1) * size(class, 1);
    end
    plot_sub_temp = [1 sub_col];
    topo_sub_temp = 1;
    plot_sub_hop = sub_col;
    topo_sub_hop = 1;
    
elseif isequal(align, 'horz')
    plot_ratio = 3;
    sub_row = num_Plot * length(chan) * size(class,1);
    sub_col = size(interval,1) * plot_ratio * size(class,1);
end

%% time-domain plot

if isequal(TimePlot, 'on')
    avgSMT = SMT;
    avgSMT = prep_selectChannels(avgSMT, {'Name', chan});
    avgSMT = prep_selectClass(avgSMT,{'class',class});
    avgSMT = prep_baseline(avgSMT, {'Time', baseline});
    avgSMT = prep_selectTime(avgSMT, {'Time', seltime});
    avgSMT = prep_average(avgSMT);

    ch_idx = find(ismember(avgSMT.chan, chan));
    
    if isfield(opt, 'TimeRange') && ~isempty(opt.TimeRange)
        time_range = opt.TimeRange;
    else
        time_range = [floor(min(reshape(avgSMT.x(:,:,ch_idx), [], 1))),...
            ceil(max(reshape(avgSMT.x(:,:,ch_idx), [], 1)))]*1.2;
    end
    for i = 1:length(chan)
        time_plt{i} = subplot(sub_row, sub_col, plot_sub_temp);
        ch_num = ismember(avgSMT.chan, chan{i});
        
        plot(avgSMT.ival, avgSMT.x(:,:,ch_num),'LineWidth',2); hold on;
        legend(avgSMT.class(:,2), 'Interpreter', 'none', 'AutoUpdate', 'off');
        %         set({'color'}, co(1:size(avgSMT.class, 1)));
        
        grid on;
        ylim(time_range);
                
        if isequal(Patch, 'on')
            % baselin patch
            base_patch = min(abs(time_range))*0.05;
            patch('XData', [baseline(1)  baseline(2) baseline(2) baseline(1)], ...
                'YData', [-base_patch -base_patch base_patch base_patch],...
                'FaceColor', 'k',...
                'FaceAlpha', 0.7, 'EdgeAlpha', 0,'faceOffsetBias', -11);
            % ival patch
            for ival = 1:size(interval,1)
                patch('XData', [interval(ival,1) interval(ival,2) interval(ival,2) interval(ival,1)],...
                    'YData', [time_range(1) time_range(1) time_range(2) time_range(2)],...
                    'FaceColor', faceColor{mod(ival,2)+1},...
                    'FaceAlpha', 0.3, 'EdgeAlpha', 0,'faceOffsetBias', -11);
            end
            
            tmp = get(time_plt{i}, 'Children');
            set(time_plt{i}, 'Children', flip(tmp));
            clear tmp;
        end
        ylabel(avgSMT.chan{ch_num}, 'Rotation', 90, ...
            'FontWeight', 'normal', 'FontSize', 12);
        hold off;
        plot_sub_temp = plot_sub_temp + sub_col;
    end
    grp_ylabel(time_plt, 'Time-Domain');
    clear avgSMT;
end
%% ERDERS plot
if isequal(ErdPlot, 'on')
    envSMT = SMT;
    envSMT = prep_selectChannels(envSMT, {'Name', chan});
    envSMT = prep_selectClass(envSMT, {'class',class});
    envSMT = prep_envelope(envSMT);
    envSMT = prep_baseline(envSMT, {'Time', baseline});
    envSMT = prep_selectTime(envSMT, {'Time', seltime});
    envSMT = prep_average(envSMT);
    
    ch_idx = find(ismember(envSMT.chan, chan));
    
    if isfield(opt, 'ERDERSRange') && ~isempty(opt.ERDERSRange)
        erders_range = opt.ERDERSRange;
    else
        erders_range = [floor(min(reshape(envSMT.x(:,:,ch_idx), [], 1))),...
            ceil(max(reshape(envSMT.x(:,:,ch_idx), [], 1)))]*1.2;
    end
        
    for i = 1:length(chan)
        erders_plt{i} = subplot(sub_row, sub_col, plot_sub_temp);
        ch_num = ismember(envSMT.chan, chan{i});
        plot(envSMT.ival, envSMT.x(:,:,ch_num),'LineWidth',2); hold on;
        legend(envSMT.class(:,2), 'Interpreter', 'none', 'AutoUpdate', 'off');

        grid on;
        
        ylim(erders_range);
                
        if isequal(Patch, 'on')
            base_patch = min(abs(erders_range))*0.05;
            % baselin patch
            patch('XData', [baseline(1)  baseline(2) baseline(2) baseline(1)], ...
                'YData', [-base_patch -base_patch base_patch base_patch],...
                'FaceColor', 'k',...
                'FaceAlpha', 0.7, 'EdgeAlpha', 0,'faceOffsetBias', -11);
            % ival patch
            for ival = 1:size(interval,1)
                patch('XData', [interval(ival,1) interval(ival,2) interval(ival,2) interval(ival,1)],...
                    'YData', [erders_range(1) erders_range(1) erders_range(2) erders_range(2)],...
                    'FaceColor', faceColor{mod(ival,2)+1},...
                    'FaceAlpha', 0.4, 'EdgeAlpha', 0,'faceOffsetBias', -11);
            end
            tmp = get(gca, 'Children');
            set(gca, 'Children', flip(tmp));
            clear tmp;
        end
        ylabel(envSMT.chan{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
        
        plot_sub_temp = plot_sub_temp + sub_col;
    end
    grp_ylabel(erders_plt, 'ERD/ERS');
    clear envSMT;
end
%% ERSP plot
if isequal(ErspPlot, 'on')
end
if isequal(FFTPlot , 'on')
    fftSMT = SMT;
    %%%%%%%
%     fft_range = [0 10];
    %%%%%%
    
    fftSMT = prep_selectChannels(fftSMT, {'Name', chan});
    fftSMT = prep_average(fftSMT);
    ch_idx = find(ismember(fftSMT.chan, chan));
    
    for i = 1:length(chan)
        for j = 1:length(class)
            fft_plt{j} = subplot(sub_row, sub_col, plot_sub_temp);
            ch_num = ismember(fftSMT.chan, chan{i});

            [YfreqDomain(j,:),frequencyRange] = positiveFFT(fftSMT.x(:,j, ch_num),fftSMT.fs);
            plot(frequencyRange,abs(YfreqDomain(j,:))); hold on;
            
            legend(fftSMT.class(j,2), 'Interpreter', 'none', 'AutoUpdate', 'off');
            
            grid on;
            
%             xlim([3 20]);
            %         ylim(fft_range);
            
            ylabel(fftSMT.chan{ch_num}, 'Rotation', 90, 'FontWeight', 'normal', 'FontSize', 12);
            
            plot_sub_temp = plot_sub_temp + sub_col;
        end
    end
    grp_ylabel(fft_plt, 'FFT');
end
%% Topo plot
if isequal(TopoPlot, 'on')
    PREVENT_OVERFLOW = 10000;
    if ~isequal(plot_sub_temp(1), 1)
        topo_sub_temp = plot_sub_temp(1);
    end
    topoSMT = SMT;    
    if size(SMT.x, 2) > PREVENT_OVERFLOW
        clear SMT;
        divis = divisors(size(topoSMT.x, 2));
        divis = divis(divis < PREVENT_OVERFLOW);
        divis = divis(end);
        SMT_size = size(topoSMT.x, 2);
        for i = 1:divis:SMT_size
            tmpSMT = prep_selectTrials(topoSMT, {'Index', [i:i+divis-1]});
            tmpSMT = prep_selectClass(tmpSMT ,{'class',class});
            if isequal(ErdPlot, 'on')
                tmpSMT = prep_envelope(tmpSMT);
            end
            tmpSMT = prep_baseline(tmpSMT, {'Time', baseline});
            topoSMT.x(:,i:i+divis-1,:) = tmpSMT.x;
        end
        clear tmpSMT;
    else
        topoSMT = prep_selectClass(topoSMT,{'class',class});
        if isequal(ErdPlot, 'on')
            topoSMT = prep_envelope(topoSMT);
        end
        topoSMT = prep_baseline(topoSMT, {'Time', baseline});
    end
    topoSMT = prep_selectTime(topoSMT, {'Time', seltime});
    topoSMT = prep_average(topoSMT);
    
    %% scalp_plot
    switch quality
        case 'high'
            resol = 256;
        case 'medium'
            resol = 144;
        case 'low'
            resol = 32;
        otherwise
            resol = 256;
    end
    
    for i = 1: size(class, 1)
        ivalSegment = size(interval,1);
        minmax = [];
        for seg = 1: ivalSegment
            SMTintervalstart = find(topoSMT.ival == interval(seg,1));
            SMTintervalEnd = find(topoSMT.ival == interval(seg,2));
            ivalSMT = squeeze(topoSMT.x(SMTintervalstart:SMTintervalEnd,i,:));
            w{i, seg} = mean(ivalSMT,1);
            minmax = [minmax; min(w{i,seg}(:)), max(w{i,seg}(:))];
        end
        %% range_options
        if ~isfloat(plot_range)
            switch plot_range
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
        for seg = 1: size(interval, 1)
            topo_plt{i,seg} = subplot(sub_row, sub_col, topo_sub_temp);
                plot_scalp(gca, w{i, seg}, MNT, p_range, resol);
            xlabel(sprintf('[%d - %d] ms',interval(seg,:)), 'FontWeight', 'normal');
            topo_sub_temp = topo_sub_temp + topo_sub_hop;
        end
        % Keeping scalp size;
        last_position = get(gca, 'Position');
        colorbar('vert');
        tmp = get(gca, 'Position');
        tmp(3:4) = last_position(3:4);
        set(gca,'Position',tmp);
        subplot(sub_row, sub_col, topo_sub_temp - size(interval,1) * topo_sub_hop);%
        set(get(gca,'Ylabel'), 'Visible','on', 'String', {upper(class{i});''}, ...
            'Interpreter', 'none', 'FontWeight', 'normal', 'FontSize', 12);
    end
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