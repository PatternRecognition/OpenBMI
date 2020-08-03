function k = kurtosis(x,excess,dim)
% kurtosis - Compute Kurtosis of a data sample
%
% Synopsis:
%   k = kurtosis(x)
%   k = kurtosis(x,excess,dim)
%   
% Arguments:
%  x: Data matrix. For vectors, compute the sample kurtosis. For
%      matrices, the result is a row vector containing the kurtosis of each
%      column. 
%  excess: Logical. If true, return the bias-corrected excess
%      kurtosis. The excess kurtosis of the normal distribution is 0,
%      whereas the proper kurtosis is 3. Default: false (standard
%      kurtosis)
%  dim: Dimension along which to operate. Default: first non-singleton
%      dimension.
%
%   Users of MATLAB STATS TOOLBOX kurtosis.m: The semantic of the
%   "flag" option is the NEGATION of the "excess" option here.
%   
% Returns:
%  k: [1 m] matrix (or in general, size of x with first non-singleton
%      dimension collapsed) of kurtosis. k(i) is the kurtosis of x(:,i)
%   
% Description:
%   The kurtosis is the fourth central moment divided by fourth power of
%   the standard deviation. Kurtosis is a measure of how outlier-prone a
%   distribution is. The Kurtosis of a normal distribution is
%   3. Distributions that are more outlier-prone than the normal
%   distribution have kurtosis greater than 3, less outlier-prone
%   distributions have kurtosis less than 3.
%
%   
% Examples:
%   randn('state', 0);kurtosis(randn([1000 1]))
%     returns 2.83 (these are samples from a normal distribution)
%   randn('state', 0);kurtosis(randn([1000 1]), 1)
%     returns the bias corrected excess kurtosis, -0.16
%   rand('state', 0);kurtosis(rand([1000 1]))
%     returns 1.8073 (the uniform distribution has smaller tails than the
%     normal distribution)
%   kurtosis(rand([100 200 300]),0,3)
%     returns the kurtosis along the third dimension.
%   
% See also: mean,std,nanmean,nanstd
% 

% Author(s), Copyright: Anton Schwaighofer, Sep 2005
% Based on kurtosis.m from the stats toolbox
% $Id: kurtosis.m,v 1.4 2007/09/04 18:17:45 neuro_toolbox Exp $

error(nargchk(1, 3, nargin));
if nargin<3,
  % Operate along the first non-singleton dimension
  dim = min(find(size(x)~=1));
  if isempty(dim), 
    dim = 1; 
  end
end
if nargin<2 | isempty(excess),
  excess = 0;
end
if size(x,dim)<2,
  k = NaN*ones(size(x));
  return
end
if excess,
  % Compute bias corrected Kurtosis:
  s = nanstd(x,0,dim);
  n = sum(~isnan(x),dim);
  ok = (n>3) & (s>0); % otherwise bias adjustment is undefined
  k = repmat(NaN, size(n));
  if any(ok)
    s = s(ok);
    n = n(ok);
    s4 = s.^4;
    m = nanmean(x,dim);
    reps = ones([1 ndims(x)]);
    reps(dim) = size(x,dim);
    m = repmat(m, reps);
    sx4 = nansum((x-m) .^ 4, dim);
    f1 = n ./ ((n-1) .* (n-2) .* (n-3) .* (s.^4));
    k(ok) = f1 .* ((n+1).*sx4(ok) - 3 * (((n-1).^3./n) .* s4));
  end
else
  m = nanmean(x,dim);
  reps = ones([1 ndims(x)]);
  reps(dim) = size(x,dim);
  m = repmat(m, reps);
  m4 = nanmean((x - m).^4,dim);
  m2 = nanmean((x - m).^2,dim);
  k = m4./m2.^2;
end
