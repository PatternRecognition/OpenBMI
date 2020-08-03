function xa= movingMedian(x, n)
%MOVINGAVERAGE - moving median
%
%Description:
% This function calculates the moving median of a matrix along
% the first dimension.

%Arguments:
% X:      input data matrix
% N:      length of window
%
% haufe 2007
%
% this code should be vectorized one day

[l ns] = size(x);

%% acausal
nh = ceil(n/2)-1;
for il = 1:l
  x_t = x(max(il-nh, 1):min(il+nh, l), :);
  xa(il, :) = nanmedian(x_t);
end

%  % causal
%  nh = n-1;
%  for il = 1:l
%    x_t = x(max(il-nh, 1):il, :);
%    xa(il, :) = nanmedian(x_t);
%  end