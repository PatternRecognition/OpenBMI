function H= ml_boxplot(x,varargin)
% ML_BOXPLOT - Display boxplots of a data sample
%
% Synopsis:
%   H= ml_boxplot(X)
%   H= ml_boxplot(X, <OPT>, 'PropertyName', PropertyValue, ...)
%   
% Arguments:
%  X: DOUBLE [d N]. Data matrix, each column is one sample of the
%      data. Boxplots will be produced for each of the d rows of this data
%      matrix
%  OPT: 
%   
% Properties:
%  notch: Logical. If true, produce a notched-box plot. Notches represent
%      a robust estimate of the uncertainty about the means for box to box
%      comparison. If false, produce a rectangular box plot. Default: 0
%  outliers: String. Symbol for outlier values (if any). Use an empty
%      string to suppress plotting outliers. Default: '+'
%  vertical: Logical. If true, produce vertical box plot, otherwise
%      horizontal. Default: 1 (vertical)
%  whiskerlength: Scalar. Defines the length of the whiskers as a
%      function of the inter-quartile ranage. If whiskerlength==0,
%      display all data values outside the box using the plotting symbol
%      'outliers'. If whiskerlength==Inf, whiskers extend to the min and
%      max of the data. Default: 1.5
%  whiskerpercentiles: [1 2] vector. These two values define
%      the inter-percentile range, that is used to determine the whiskers, 
%      e.g., the upper whisker is
%          wp(2) + (wp(2)-wp(1))*whiskerlength, 
%      where wp are the percentile values corresponding to 
%      'whiskerpercentiles'. Default: [25 75]
%  quantilesgiven: Logical. If true, assume that each row contains 5
%      values, namely [min q25 median q75 max] of the original
%      data. This is helpful to produce several boxplots if data have
%      different length. Here, whiskers will extend to min and max of the
%      data, instead of the default behaviour with whiskerlength depending
%      on the inter-quartile range. Default: 0.
%  boxposition: [d 1] vector. X-position (for vertical plots) resp Y-position
%      of the boxplot for each row. If this argument is supplied, no axis
%      labelling will be produced. This is helpful to put boxplot at
%      arbitrary positions on the figure. Default: 1:d
%  boxwidth: Scalar. Width of the box. A default is computed depending on
%      the number of boxplots.
%
% Returns:
%  H: Struct array of handles to graphical objects.
%
% Description:
%   This routine produces a box and whisker plot for each row of X. The
%   box has lines at the lower quartile, median, and upper quartile
%   values. The whiskers are lines extending from each end of the box to
%   show the extent of the rest of the data.  Outliers are data with values
%   beyond the ends of the whiskers.
%
%   
% Examples:
%   boxplot([1 2 3; 4 5 10])
%     will produce two boxplots for the first and second row of the data.
%   boxplot([1 2 3 10 15], 'quantilesgiven', 1)
%     will produce a boxplot with minimum 1, median 3, maximum 15, and
%     quartiles 2 and 10.
%   boxplot([1 2 3 10 15], 'quantilesgiven', 1, 'boxposition', 3.14, 'boxwidth',0.3)
%     will produce a boxplot at x-coordinate 3.14 with a box width of 0.3.
%   H= boxplot(randn(2,100));
%   set(struct2array(H), 'LineWidth',2);
%   set([H(1).box H(1).whiskers H(1).outliers], 'Color','m'); 
%   set([H(2).box H(2).whiskers H(2).outliers], 'Color',[0.9 0.7 0]);
%   set([H.whiskers], 'LineStyle',':');
%     will give the boxes an individual look
%   
% See also: percentiles

% Author(s), Copyright: Anton Schwaighofer, May 2005
% $Id: boxplot.m,v 1.8 2007/09/12 07:56:09 neuro_toolbox Exp $


defspec=  {'MarkerFaceColor',[1 0.9 0.4], ...
           'MarkerSize', 3};
opt= propertylist2struct(varargin{:});
[opt, isdefault]= set_defaults(opt, 'notch', 0, ...
                                    'outliers', '+', ...
                                    'vertical', 1, ...
                                    'whiskerpercentiles', [25 75], ...
                                    'whiskerlength', 1.5, ...
                                    'quantilesgiven', 0, ...
                                    'boxposition', [], ...
                                    'boxwidth', [], ...
                                    'percentiles', [0 25 50 75 100], ...
                                    'use_median', 0, ...
                                    'plot_distribution', 0, ...
                                    'distribution_spec', defspec);
%                                    'use_prctile', 0, ...

%if opt.use_prctile,
%  percentiles= @prctile;
%end

if opt.plot_distribution,
  if iscell(x),
    data= x;
  else
    data= num2cell(x, 2);
  end
end

if iscell(x),
  if numel(x)>length(x),
    error('Cannot handle multi-dimensional cell arrays');
  end
  if ~isdefault.whiskerpercentiles | ~isdefault.whiskerlength,
    warning('When data is given as cell, whisker definitions are ignored.');
  end
  nCells= length(x);
  perc= zeros(nCells, 5);
  for c= 1:nCells,
    perc(c,:)= percentiles(x{c}, opt.percentiles);
  end
  x= perc;
  opt.quantilesgiven= 1;
end

if opt.quantilesgiven,
  if opt.notch,
    warning('Notched boxplots can not be computed from given quantiles');
  end
  if size(x,2)~=5,
    error('With the ''quantilesgiven'' option, data must have 5 columns');
  end
end

if min(size(x)) == 1,
  % If vector given as input, always assume that this is the data
  x = x(:)';
end
[dim, N] = size(x);

positionGiven = 1;
if isempty(opt.boxposition),
  positionGiven = 0;
  opt.boxposition = 1:dim;
elseif length(opt.boxposition)~=dim,
  error('Option ''boxposition'' must be a vector of length d');
end

xlims = [min(opt.boxposition)-0.5 max(opt.boxposition)+0.5];

k = find(~isnan(x));
ymin = min(min(x(k)));
ymax = max(max(x(k)));
dy = (ymax-ymin)/20;
ylims = [(ymin-dy) (ymax+dy)];

if isempty(opt.boxwidth),
  opt.boxwidth = (max(xlims) - min(xlims)) * min(0.15,0.5/dim);
end

% Scale axis for vertical or horizontal boxes, but only if no plot
% position is supplied
if ~positionGiven,
  if opt.vertical
    axis([xlims ylims]);
    set(gca,'XTick', opt.boxposition);
    set(gca,'YLabel',text(0,0,'Values'));
    set(gca,'XLabel',text(0,0,'Row Number'));
  else
    axis([ylims xlims]);
    set(gca,'YTick', opt.boxposition);
    set(gca,'XLabel',text(0,0,'Values'));
    set(gca,'YLabel',text(0,0,'Row Number'));
  end
end

holdState = get(gca,'NextPlot');
set(gca,'NextPlot','add');
for i=1:dim,
  z = x(i,:);
  vec = find(~isnan(z));
  if opt.plot_distribution,
    yy= data{i};
    [dmy, dmy, pdf]= cheap_pdf1(yy);
    pdf= pdf/max(pdf);
    xx= pdf.*(rand(1, length(yy))-ones(1,length(yy))/2) * opt.boxwidth;
    Hd(i).distribution= plot(opt.boxposition(i) + xx, yy, 'o', ...
                             'MarkerEdgeColor', 'none', ...
                             opt.distribution_spec{:});
  end
  if ~isempty(vec)
    H(i)= boxutil(z(vec),opt.boxposition(i),opt);
  end
end
if opt.plot_distribution,
  H= merge_structs(H, Hd);
end

set(gca,'NextPlot',holdState);
if nargout==0,
  clear H
end


%% ---- subfunction for boxplotting one vector

function H= boxutil(x,lb,opt)
% produces a single box plot.

%if opt.use_prctile,
%  percentiles= @prctile;
%end

lf = opt.boxwidth;

% Make sure X is a vector.
if min(size(x)) ~= 1, 
    error('First argument has to be a vector.'); 
end

% define the median and the quantiles
if opt.use_median,
  med = median(x);  % percentiles(x,50) does not inpolate for even # of samples
else
  med = percentiles(x,50);
end
q3 = percentiles(x,75);

%% what is this nonsense with 40% percentile??
%% ---- begin old code [buggy: botplot(1:5, 'quantilesgiven',1)
% $$$ if opt.quantilesgiven,
% $$$   % 40% percentile gives exactly the second position of the ordered data
% $$$   q1 = percentiles(x,40);
% $$$   % Make whiskers extend to min and max instead of IQR.
% $$$   opt.whiskerlength = inf;
% $$$ else
% $$$   q1 = percentiles(x,25);
% $$$ end
%% ---- end old code

%% ---- begin new code
q1 = percentiles(x,25);
if opt.quantilesgiven,
  % Make whiskers extend to min and max instead of IQR.
  opt.whiskerlength = inf;
end
%% ---- end new code

% find the extreme values (to determine where whiskers appear)
whiskerperc= percentiles(x, opt.whiskerpercentiles);
vhi = whiskerperc(2) + opt.whiskerlength * diff(whiskerperc);
upadj = max(x(x<=vhi));
if (isempty(upadj)), upadj = whiskerperc(2); end

vlo = whiskerperc(1) - opt.whiskerlength * diff(whiskerperc);
loadj = min(x(x>=vlo));
if (isempty(loadj)), loadj = whiskerperc(1); end

x1 = lb*ones(1,2);
x2 = x1+[-0.25*lf,0.25*lf];
% The outliers:
yy = x(x<loadj | x > upadj);
% Plot position for the outliers
xx = lb*ones(1,length(yy));

lbp = lb + 0.5*lf;
lbm = lb - 0.5*lf;

% Set up (X,Y) data for notches if desired.
if ~opt.notch
    xx2 = [lbm lbp lbp lbm lbm];
    yy2 = [q3 q3 q1 q1 q3];
    xx3 = [lbm lbp];
else
    n1 = med + 1.57*(q3-q1)/sqrt(length(x));
    n2 = med - 1.57*(q3-q1)/sqrt(length(x));
    if n1>q3, n1 = q3; end
    if n2<q1, n2 = q1; end
    lnm = lb-0.25*lf;
    lnp = lb+0.25*lf;
    xx2 = [lnm lbm lbm lbp lbp lnp lbp lbp lbm lbm lnm];
    yy2 = [med n1 q3 q3 n1 med n2 q1 q1 n2 med];
    xx3 = [lnm lnp];
end
yy3 = [med med];

% Determine if the boxes are vertical or horizontal.
% The difference is the choice of x and y in the plot command.
if opt.vertical
  H.box= plot(xx2, yy2, 'b-');
  H.whiskers= plot(x1,[q3 upadj], 'b--', x1,[loadj q1], 'b--')';
  H.whisker_ends= plot(x2,[loadj loadj],'k-', ...
                       x2,[upadj upadj],'k-')';
  H.median= plot(xx3,yy3,'r-');
  % Plot outliers only if not suppressed by empty outliers symbol
  if ~isempty(yy) & ~isempty(opt.outliers),
    H.outliers= plot(xx, yy, opt.outliers);
  else
    H.outliers= [];
  end
else
  H.whiskers= plot([q3 upadj],x1,'b--',[loadj q1],x1,'b--')';
  H.whisker_ends= plot([loadj loadj],x2,'k-',...
                       [upadj upadj],x2,'k-')';
  H.box= plot(yy2,xx2,'b-');
  H.median=  plot(yy3,xx3,'r-');
  % Plot outliers only if not suppressed by empty outliers symbol
  if ~isempty(yy) & ~isempty(opt.outliers),
    H.outliers= plot(yy, xx, opt.outliers);
  else
    H.outliers= [];
  end
end

