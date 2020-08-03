function [xquant,yquant,ind] = quantileplot(r,varargin)
% quantileplot - Quantile-quantile-plot to check for normality
%
% Synopsis:
%   quantileplot(X)
%   [xquant,yquant,ind] = quantileplot(X,'Property',Value,...)
%   
% Arguments:
%  X: [1 N] matrix of data
%   
% Returns:
%  xquant: [1 N] matrix. Quantiles of the standard normal distribution
%      N(0,1)
%  yquant: [1 N] matrix. Quantiles of the data X
%  ind: [1 N] matrix. For plotting, data X will be sorted. Return
%      argument ind is the sorting vector (the second return arg of
%      sort.m)
%   
% Properties:
%  plot: Logical. If 1, plot yquant versus xquant. Defaults to 1 if no
%      output argument given
%  xlabel: Label for x-axis, default: 'Quantiles of N(0,1)'
%  ylabel: Label for y-axis, default: 'Data quantiles'
%  marker: Cell array. Marker for plotting. Default: {'k.', 'MarkerSize',
%      6'}
%  diagonal: Cell array. If non-empty, do also plot the diagonal line. 
%      Default value: {'b-', 'Linewidth', 1}
%   
% Description:
%   A quantile-quantile plot can be used to check whether data follows a
%   normal distribution. It plots quantiles of the data versus quantiles
%   of the standard normal distribution. If the data is normal
%   distributed, the plots should follow a diagonal line.
%   The routine here assumes that the data X follow an N(0,1)
%   distribution, no further scaling is done before plotting.
%   
%   
% Examples:
%   Plot resdiuals from a model fit:
%     r = y-pred;
%     quantileplot(r/std(r));
%   (Residuals should already have mean 0, only need to scale)
%   
% See also: normal_invcdf
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2005
% $Id: quantileplot.m,v 1.2 2005/09/02 14:56:16 neuro_toolbox Exp $

error(nargchk(1, inf,nargin))
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'plot', nargout==0, ...
                        'xlabel', 'Quantiles of N(0,1)', ...
                        'ylabel', 'Data quantiles', ...
                        'marker', {'k.', 'MarkerSize',6}, ...
                        'diagonal', {'b-', 'Linewidth', 1});

r = r(:)';
N = length(r);
xquant = normal_invcdf(((1:N)-0.5)/N);
[yquant,ind] = sort(r);
if opt.plot,
  % Get a common range for plotting, so that the diagonal is visible
  m1 = min(min(xquant, yquant));
  m2 = max(max(xquant, yquant));
  s = get(gca, 'NextPlot');
  if ~isempty(opt.diagonal),
    plot([m1 m2], [m1 m2], opt.diagonal{:});
    hold on;
  end
  plot(xquant, yquant, opt.marker{:});
  axis([m1 m2 m1 m2]);
  xlabel(opt.xlabel);
  ylabel(opt.ylabel);
  set(gca, 'NextPlot', s);
end
% Clear the variable if no output arguments requested. Otherwise, calling
% without trailing semicolon would print out the contents of xquant
if nargout==0,
  clear xquant;
end
