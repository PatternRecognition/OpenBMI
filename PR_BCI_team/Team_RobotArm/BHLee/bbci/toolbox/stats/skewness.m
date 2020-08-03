function s= skewness(x, no_bias_adjustment)
% skewness - Compute Skewness of a data sample
%
% Synopsis:
%   s = skewness(x)
%   s = skewness(x, no_bias_adjustment)
%
% Arguments:
%  x: [n m] matrix. For vectors, compute the sample skewness. For
%      matrices, the result is a row vector containing the skewness of each
%      column. In general, skewness operates along the first non-singleton
%      dimension.
%  no_bias_adjustment: Logical. If true, return the uncorrected 
%      skewness. Default: true (standard skewness).
%
% Returns:
%  k: [1 m] matrix (or in general, size of x with first non-singleton
%      dimension collapsed) of skewness. k(i) is the skewness of x(:,i).
%
% Description:
%   The skewness is the third central moment divided by the cube of
%   the standard deviation. It is a measure of how asymmetric a
%   distribution is.
%
% Examples:
%   randn('state', 0); skewness(randn([1000 1]))
%     returns -0.030682  %% normal distribution is symmetric
%   randn('state', 0); skewness(randn([1000 1]).^2)
%     returns 2.3227     %% but not when squared
%
% See also: mean, std, kurtosis, nanmean, nanstd
%

if nargin<2,
  no_bias_adjustment= 1;
end

if ~no_bias_adjustment,
  error('Not implemented. Buy stats toolbox.');
end

szx= size(x);
dim= max(1, min(find(szx~=1)));
rep= ones(size(szx));
rep(dim)= szx(dim);
m = nanmean(x, dim);
m = repmat(m, rep);
m3 = nanmean((x - m).^3, dim);
sm2 = sqrt(nanmean((x - m).^2, dim));
s = m3./sm2.^3;
