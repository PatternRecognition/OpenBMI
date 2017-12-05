function output = visual_scalpPlot_fin(SMT, varargin)
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
subplot = @(m,n,p) subtightplot(m,n,p,[0.045 0]);

figure;
set(gcf,'Position',[400 200 1000 600]);
MNT = opt_getMontage(SMT);

opt = opt_cellToStruct(varargin{:});
if isfield(opt, 'Interval') interval = opt.Interval; end
if isfield(opt, 'Range') p_range = opt.Range; end
if isfield(opt, 'Resolution') resol = opt.Resolution; else resol = 300; end
if isfield(opt, 'Channels') chan = opt.Channels; end
if isfield(opt, 'Class') class = opt.Class; end
if isfield(opt, 'Color') faceColor = opt.Color; else faceColor = [{[0.8 0.8 0.8]};{[0.6 0.6 0.6]}]; end
if isfield(opt, 'TimePlot') TimePlot = opt.TimePlot; else TimePlot = 'on'; end
if isfield(opt, 'TimePlot') TopoPlot = opt.TopoPlot; else TopoPlot = 'on'; end
if isfield(opt, 'Baseline') baseline = opt.Baseline; else baseline = [0 0]; end

output_str = 'Finished';

if ~isequal(MNT.chan, SMT.chan)
    if length(SMT.chan) > length(MNT.chan)
        tmp = SMT.chan(~ismember(SMT.chan, MNT.chan));
        output_str = tmp{1};
        for i = 2:length(tmp)
            output_str = [output_str sprintf(', %s',tmp{i})];
        end
        output_str = [output_str, ' are missed'];
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
% 
SMT = prep_selectClass(SMT,{'class',class});
avgSMT= prep_average(SMT);

p_range = [-1.5 0.5];
oldUnit = get(gcf,'units');
set(gcf,'units','normalized');

sub_row = length(chan) + size(SMT.class,1);
sub_col = size(interval,1);

%% time-domain plot
plot_position = 1;

ch_idx = find(ismember(SMT.chan, chan));
time_range = [floor(min(reshape(avgSMT.x(:,:,ch_idx), [], 1))),...
    ceil(max(reshape(avgSMT.x(:,:,ch_idx), [], 1)))];

if isequal(TimePlot, 'on')
    for i = 1:length(chan)
        subplot(sub_row, sub_col, plot_position:plot_position + sub_col -1);
        ch_num = ismember(SMT.chan, chan{i});
        
        for class = 1:size(SMT.class,1)
            plot(SMT.ival, avgSMT.x(:,class,ch_num),'LineWidth',2); hold on;
        end
        
        grid on;
        ylim(time_range);
        
        for cl_num=1:length(SMT.class(:,2))
            i_legend{cl_num}=char(SMT.class(cl_num,2));
        end
        
        legend(i_legend, 'Interpreter', 'none', 'AutoUpdate', 'off');
        
        for ival = 1:size(interval,1)
            patch('XData', [interval(ival,1) interval(ival,2) interval(ival,2) interval(ival,1)],...
                'YData', [time_range(1) time_range(1) time_range(2) time_range(2)],...
                'FaceColor', faceColor{mod(ival,2)+1},...
                'FaceAlpha', 0.7, 'EdgeAlpha', 0,'faceOffsetBias', -11);
        end
        
        tmp = get(gca, 'Children');
        set(gca, 'Children', flip(tmp));
        clear tmp;
        
        ylabel(sprintf('%s',SMT.chan{ch_num}), 'Rotation', 0, 'HorizontalAlignment','right');
        
        plot_position = plot_position + sub_col;
    end
end
if isequal(TopoPlot, 'on')
    %% scalp_plot
    for i = 1: size(SMT.class,1)
        for seg = 1: size(interval, 1)
            subplot(sub_row, sub_col, plot_position);
            SMTintervalstart = find(avgSMT.ival == interval(seg,1));
            SMTintervalEnd = find(avgSMT.ival == interval(seg,2))-1;
            
            ivalSMT = avgSMT.x(SMTintervalstart:SMTintervalEnd,:,:);
            w = mean(squeeze(ivalSMT(:,i,:)),1);
            
            scalp_plot(gca, w, MNT, p_range, resol);
            plot_position = plot_position + 1;
        end
        %     subplot(size(SMT.class,1),size(opt.Ival,2),plot_position)
        % Keeping scalp size;
        last_position = get(gca, 'Position');
        colorbar('vert');
        tmp = get(gca, 'Position');
        tmp(3:4) = last_position(3:4);
        set(gca,'Position',tmp);
        %     axis off;
        %     plot_position = plot_position + 1;
    end
end
output = output_str;
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