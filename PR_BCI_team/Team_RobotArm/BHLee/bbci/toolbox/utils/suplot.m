function h= suplot(n, j, varargin)
%h= suplot(n, j, params)
%
% n: number of plots to be placed on the figure
% j: number current subplot
% params: if params are given, subplotxl is called instead of subplot

qn= floor(sqrt(n));
rn= ceil(n/qn);
if nargin>2,
  h= subplotxl(qn, rn, j, varargin{:});
else
  h= subplot(qn, rn, j);
end
