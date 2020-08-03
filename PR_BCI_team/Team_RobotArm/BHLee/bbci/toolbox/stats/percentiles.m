function y = percentiles(x,p)
% percentiles - Percentiles of a data sample
%
% Synopsis:
%   y = percentiles(X,p)
%   
% Arguments:
%  X: [d N] matrix. Data matrix, each column is one sample of the
%     data. Row or column vectors are treated as if the were of size [1 N]
%  p: [1 m] vector, with entries in the range 0..100. Produce
%      percentile values for the values given here. For scalar p, y is a row
%      vector containing Pth percentile of each column of X. For vector p,
%      the ith row of y is the p(i) percentile of each column of X.
%   
% Returns:
%  y: [d m] matrix. Percentile values for each row of the data X. 
%     y(i,j) is the P(j) percentile of the data row X(i,:)
%   
% Examples:
%   y = percentiles([1 3 100], 50)
%     returns the median of the vector, 3.
%   y = percentiles([1 3 100; 4 5 6], [50 80])
%     returns median and 80% percentiles of each row:
%         3   100
%         5     6
%
% See also: boxplot
% 

% Author(s), Copyright: Anton Schwaighofer, May 2005
% $Id: percentiles.m,v 1.4 2006/08/30 14:21:27 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

if min(size(x))==1,
  x = x(:)';
end
if min(size(p))>1,
  error('P must be a scalar or a vector.');
end
if any(p > 100) | any(p < 0)
  error('P must take values between 0 and 100');
end
p = p(:)/100;

d = size(x, 1);
y = zeros(d, length(p));

%% If x does not contain NaN elements, and has more than one row,
%% this way the vector 'pos' is calculated for each row.
for i = 1:d,
  xi = x(i,:);
  xi = xi(~isnan(xi));
  xi = sort(xi);
  N= size(xi, 2);      %% number of ~isnan elements
  
  % This is the somewhat obvious, but not correct way
  % pos= 1 + round(p*(N-1));

  % [check: try percentiles(1:5, 0:25:100)]
  pos = zeros([1 length(p)]);
  %% Correct way of rounding: Towards end (ceil) for anything larger than
  %% 0.5, towards beginning (floor) for anything below 0.5
  pos(p>=.5) = ceil(p(p>=.5)*N);
  pos(p<.5) = 1+floor(p(p<.5)*N);

  y(i,:) = xi(pos);
end
