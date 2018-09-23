function out = vis_subplotTemplate(opt)
% -----------------------------------------------------
% FileExchange function subtightplot by F. G. Nievinski
subplot = @(m,n,p) subtightplot(m,n,p,[0.045 0.01], 0.05, [0.06, 0.048]);

def_opt = struct('align', 'vert', 'timeplot', 'off', 'topoplot', 'off',...
    'fftplot', 'off', 'rvalue', 'off');

opt = opt_defaultParsing(def_opt, opt);

RATIO = 2;

channels = length(opt.channels);
class = size(opt.class, 1);
interval = size(opt.interval, 1);

n_fp = channels * class * strcmpi(opt.fftplot, 'on');
n_tp = channels * strcmpi(opt.timeplot, 'on');
n_rp = double(strcmpi(opt.rvalue, 'on'));
n_topo_row = (class + n_rp) * strcmpi(opt.topoplot, 'on');
n_topo_col = interval * strcmpi(opt.topoplot, 'on');

time_plt = gobjects(1,n_tp);
fft_plt = gobjects(1,n_fp);
topo_plt = gobjects(1,(n_topo_row - n_rp)*n_topo_col);
r_plt = gobjects(1, n_rp * n_topo_col + n_rp *  strcmpi(opt.timeplot, 'on'));

monitor_screensize = get(0, 'screensize');

fig = figure('Color', 'w');
set(fig, 'ToolBar', 'none');


if strcmpi(opt.align, 'vert') || xor(sum([n_fp, n_tp]), n_topo_row * n_topo_col)
    set(gcf,'Position',[monitor_screensize(3)/4,  monitor_screensize(4)/20,...
        monitor_screensize(3)/2, monitor_screensize(3)/2]);
    sr = n_tp + n_fp + n_topo_row + n_rp * strcmpi(opt.timeplot, 'on');
    sc = max(n_topo_col, 1);
    %% time
    template = [1 sc];
    if isequal(opt.timeplot, 'on')
        for i = 1:n_tp
            time_plt(i) = subplot(sr, sc, template);
            template = template + sc;
        end
    end
    %% fft
    if isequal(opt.fftplot, 'on')
        for i = 1: n_fp
            fft_plt(i) = subplot(sr, sc, template);
            template = template + sc;
        end
    end
    %% topo
    if isequal(opt.topoplot, 'on')
        for i = 1:(n_topo_row-n_rp) * n_topo_col
            topo_plt(i) = subplot(sr, sc, template(1));
            template = template + 1;
        end
    end
    %% r-value
    % TODO: multiclass rplots...
    if isequal(opt.rvalue, 'on')
        if isequal(opt.topoplot, 'on')
            for i = 1:n_rp * n_topo_col
                r_plt(i) = subplot(sr, sc, template(1));
                template = template + 1;
            end
        end
        if isequal(opt.timeplot, 'on')
            r_plt(n_rp * n_topo_col + 1) = subplot(sr, sc, template);
            template = template + sc;
        end
    end
    
elseif strcmpi(opt.align, 'horz')
    set(gcf,'Position',[monitor_screensize(3)/4,  monitor_screensize(4)/4,...
        monitor_screensize(3)/2, monitor_screensize(4)/2]);
    
    sr = (n_tp + num_ep + n_fp) * n_topo_row;
    sc = (1 + RATIO) * n_topo_col + 2;
    
    if isequal(opt.RPlot, 'on')
        sc = sc + n_topo_col + 2;
    end
    %% graph
    template = [1 (n_topo_row-1)*sc+n_topo_col];
    for i = 1:n_tp + num_ep + n_fp
        time_plt(i) = subplot(sr, sc, template);
        template = template + n_topo_row*sc;
    end
    %% topo
    template = [n_topo_col+3 (sr/class-1)*sc+RATIO-1+n_topo_col+3];
    for i = 1:n_topo_row
        for j = 1:n_topo_col
            topo_plt((i-1)*n_topo_col +  j) = subplot(sr, sc, template);
            template = template + RATIO;
        end
        template = template + sr/class*sc - RATIO*n_topo_col;
    end
    %% r-value
    if isequal(opt.rvalue, 'on')
        template = [sc-n_topo_col+1 n_topo_row*sc];
        for i = 1:n_tp + num_ep + n_fp
            r_plt(i) = subplot(sr, sc, template);
            template = template + n_topo_row*sc;
        end
    end
end

out = horzcat(time_plt, fft_plt, topo_plt, r_plt);
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