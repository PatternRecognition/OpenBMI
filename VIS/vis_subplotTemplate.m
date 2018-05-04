function out = vis_subplotTemplate(opt)
% -----------------------------------------------------
% FileExchange function subtightplot by F. G. Nievinski
subplot = @(m,n,p) subtightplot(m,n,p,[0.045 0.01], 0.05, [0.06, 0.048]);
% subplot = @(m,n,p) subtightplot(m,n,p);

if ~isfield(opt, 'FFTPlot')
    opt.FFTPlot = 'off';
end
if ~isfield(opt, 'TimePlot')
    opt.TimePlot = 'off';
end
% if ~isfield(opt, 'ErdPlot')
%     opt.ErdPlot = 'off';
% end
if ~isfield(opt, 'rValue')
    opt.rValue = 'off';
end
if ~isfield(opt, 'TopoPlot')
    opt.TopoPlot = 'off';
end
if ~isfield(opt, 'Align')
    opt.Align = 'vert';
end

RATIO = 2;

channels = length(opt.Channels);
class = size(opt.Class, 1);
interval = size(opt.Interval, 1);

num_fp = channels * class * strcmpi(opt.FFTPlot, 'on');
num_tp = channels * strcmpi(opt.TimePlot, 'on');
% num_ep = channels * strcmpi(opt.ErdPlot, 'on');
num_rp = double(strcmpi(opt.rValue, 'on'));
num_topo_row = (class + num_rp) * strcmpi(opt.TopoPlot, 'on');
num_topo_col = interval * strcmpi(opt.TopoPlot, 'on');

% graph_plt = gobjects(1,sum([num_fp,num_tp, num_ep]));
time_plt = gobjects(1,num_tp);
fft_plt = gobjects(1,num_fp);
topo_plt = gobjects(1,(num_topo_row - num_rp)*num_topo_col);
r_plt = gobjects(1, num_rp * num_topo_col + num_rp *  strcmpi(opt.TimePlot, 'on'));
%
% graph_plt = [];
% topo_plt = [];
% r_plt = [];

fig = figure('Color', 'w');
set(fig, 'ToolBar', 'none');

monitor_screensize = get(0, 'screensize');

if strcmpi(opt.Align, 'vert') || xor(sum([num_fp, num_tp]), num_topo_row * num_topo_col)
    set(gcf,'Position',[monitor_screensize(3)/4,  monitor_screensize(4)/20,...
        monitor_screensize(3)/2, monitor_screensize(3)/2]);
    
%     sr = num_tp + num_ep + num_fp + num_topo_row + num_rp;
    sr = num_tp + num_fp + num_topo_row + num_rp * strcmpi(opt.TimePlot, 'on');
    sc = max(num_topo_col, 1);
    %% graph
    template = [1 sc];
    if isequal(opt.TimePlot, 'on')
        for i = 1:num_tp
            time_plt(i) = subplot(sr, sc, template);
            template = template + sc;
        end
    end
    
    if isequal(opt.FFTPlot, 'on')
        for i = 1: num_fp
            fft_plt(i) = subplot(sr, sc, template);
            template = template + sc;
        end
    end
    %% topo
    if isequal(opt.TopoPlot, 'on')
        template = template(1);
        for i = 1:(num_topo_row-num_rp) * num_topo_col
            topo_plt(i) = subplot(sr, sc, template);
            template = template + 1;
        end
    end
    %% r-value
    if isequal(opt.rValue, 'on')
        if isequal(opt.TopoPlot, 'on')
            template = template(1);
            for i = 1:num_rp * num_topo_col
                r_plt(i) = subplot(sr, sc, template);
                template = template + 1;
            end
        end
        %%TODO::
        if isequal(opt.TimePlot, 'on')
            template = [template(1) template(1)+sc-1];
            r_plt(max([num_rp * num_topo_col,1])) = subplot(sr, sc, template);
            template = template + sc;
        end
    end
    
elseif strcmpi(opt.Align, 'horz')
    set(gcf,'Position',[monitor_screensize(3)/4,  monitor_screensize(4)/4,...
        monitor_screensize(3)/2, monitor_screensize(4)/2]);
    
    sr = (num_tp + num_ep + num_fp) * num_topo_row;
    sc = (1 + RATIO) * num_topo_col + 2;
    
    if isequal(opt.RPlot, 'on')
        sc = sc + num_topo_col + 2;
    end
    %%
    %     test(sr, sc);
    %% graph
    template = [1 (num_topo_row-1)*sc+num_topo_col];
    for i = 1:num_tp + num_ep + num_fp
        time_plt(i) = subplot(sr, sc, template);
        template = template + num_topo_row*sc;
    end
    %% topo
    template = [num_topo_col+3 (sr/class-1)*sc+RATIO-1+num_topo_col+3];
    for i = 1:num_topo_row
        for j = 1:num_topo_col
            topo_plt((i-1)*num_topo_col +  j) = subplot(sr, sc, template);
            template = template + RATIO;
        end
        template = template + sr/class*sc - RATIO*num_topo_col;
    end
    %% r-value
    if isequal(opt.RPlot, 'on')
        template = [sc-num_topo_col+1 num_topo_row*sc];
        for i = 1:num_tp + num_ep + num_fp
            r_plt(i) = subplot(sr, sc, template);
            template = template + num_topo_row*sc;
        end
    end
end

out = horzcat(time_plt, fft_plt, topo_plt, r_plt);
end

function test(sr, sc)
subplot = @(m,n,p) subtightplot(m,n,p,[0.045 0.01], 0.05, [0.06, 0.048]);
for i = 1:sr*sc
    subplot(sr, sc, i);
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