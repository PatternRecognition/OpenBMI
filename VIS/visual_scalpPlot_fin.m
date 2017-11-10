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
% file = fullfile('D:\OPENBMI_NEW_PROJECT\Topographic\ORIGIN_BBCI\data', '2017_10_27_yelee_exp2');
% file = fullfile('D:\StarlabDB_2nd\subject9_mhlee\session1', 'p300_off');
% time = [-200 1000]; baseline = [-200 0]; freq = [5 40];
% field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
% marker = {'1', 'target'; '2', 'non_target'};
% % marker = {'11', 'N_target'; '12', 'N_nontarget'; '21','P_target'; '22',...
% %     'P_nontarget'; '31','C_target'; '32', 'C_nontarget'};
% % marker = {'11', 'target'; '12', 'non_target'};
% [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs', 1000});
% CNT = opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% CNT = prep_filter(CNT, {'frequency', freq});
% SMT = prep_segmentation(CNT, {'interval', time});
% SMT = prep_baseline(SMT, {'Time',baseline});
% load epo1;
% SMT.x = epo1.x;
% SMT.x = reshape(SMT.x, 600, 720, 19);

%% FileExchange function subtightplot by F. G. Nievinski
subplot = @(m,n,p) subtightplot(m,n,p); 
%% -----------------------------------------------------
MNT = opt_getMontage(SMT);
if ~isequal(MNT.chan, SMT.chan)
    if length(SMT.chan) > length(MNT.chan)
        warning('Some channels are missed');
    end
    tmp = zeros(1,length(MNT.x));
    for i = 1:length(MNT.chan)
        tmp(i) = find(ismember(SMT.chan, MNT.chan{i}));
    end
    SMT.x = SMT.x(:,:,tmp);
    SMT.chan = SMT.chan(tmp);
    clear tmp;
end
opt = opt_cellToStruct(varargin{:});
opt.Interval = abs(opt.Ival(1)-opt.Ival(2));

xe = MNT.x';
ye = MNT.y';
avgSMT= prep_average(SMT);

line_width = 1;
resolution = 300;
p_range = [-1.5 0.5];
% sub_ax = subaxes(size(SMT.class,1),size(opt.Ival,2)-1,get(gcf,'DefaultAxesPosition'),gcf);

% colormap(jet(51));
% cm = get(gcf, 'Colormap');
% colormap(cm);
for i = 1: size(SMT.class,1)
    for seg = 1: size(opt.Ival, 2)-1
        %     figure()
        subplot(size(SMT.class,1),size(opt.Ival,2)-1,((i-1)*(size(opt.Ival, 2)-1))+seg)

        center = [0 0];
        theta = linspace(0,2*pi,360);
        
        x = cos(theta)+center(1);
        y = sin(theta)+center(2);
        
        SMTintervalstart = find(avgSMT.ival == opt.Ival(seg))-1;
        SMTintervalEnd = find(avgSMT.ival == opt.Ival(seg) + opt.Interval)-1;
        UpdatedSMT{seg} = avgSMT.x(SMTintervalstart:SMTintervalEnd,:,:);
        w{seg,i} = UpdatedSMT{seg}(:,i,:);
        w{seg,i} = squeeze(w{seg,i});
        inputx{seg,i}  = mean(w{seg,i},1);
        
        
        maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
        
        xx = linspace(-maxrad, maxrad, resolution);
        yy = linspace(-maxrad, maxrad, resolution)';  
        
        oldUnit = get(gcf,'units');
        set(gcf,'units','normalized');
        
        H = struct('ax', gca);
        set(gcf,'CurrentAxes',H.ax);
        
        % ----------------------------------------------------------------------
        % contour plot
%         F = scatteredInterpolant(xe, ye, inputx{seg,i}');
%         a = F(xx', yy);
        [xg,yg,zg] = griddata(xe, ye, inputx{seg,i}, xx, yy, 'v4');
        mask = ~(sqrt(xg.^2+yg.^2)<=maxrad);
        zg(mask)=NaN;
        
        contourf(xg, yg, zg, 50, 'LineStyle','none');
        hold on;
        
        patch([0 0], [0 0], [1 2]);
        ccc = get(gca, 'children');
        set(ccc(1), 'Visible', 'off');
        
        if diff(p_range)==0, p_range(2)= p_range(2)+eps; end
        
        set(gca, 'CLim', p_range);
        % ----------------------------------------------------------------------
        % contour line
        p_min = p_range(1);
        p_max = p_range(2);
        
        cl = linspace(p_min, p_max, 7);
        cl = cl(2:end-1);
        
        contour(xg, yg, zg, cl, 'k-');
        
        % ----------------------------------------------------------------------
        % disp electrodes
        plot(xe, ye, 'k.'); hold on;
        set(0,'defaultfigurecolor',[1 1 1])
        
        % ----------------------------------------------------------------------
        % nose plot
        nose = [1 1.2 1];
        nosi = [83 90 97]+1;
        H.nose = plot(nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', line_width );
        
        hold on;
        
        % ----------------------------------------------------------------------
        % ears plot
        earw = .08; earh = .3;
        H.ears(1) = plot(x*earw-1-earw, y*earh, 'k', 'linewidth', line_width);
        H.ears(2) = plot(x*earw+1+earw, y*earh, 'k', 'linewidth', line_width);
        hold on;
        
        % ----------------------------------------------------------------------
        % main circle plot
        H.main = plot(x,y, 'k');
        set(H.ax, 'xTick',[], 'yTick',[]);
        axis('xy', 'tight', 'equal', 'tight');
        hold on;
        
        axis off;
        
        
        title({[SMT.class{i,2},' class'];['[' , num2str(opt.Ival(seg)), ' ~ ' , num2str(opt.Ival(seg+1)) , '] ms']})
        %         colorbar('vert');
    end
end
output = gcf;
end

%% Function scalp_plot
function scalp_plot(ax, data, MNT)

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