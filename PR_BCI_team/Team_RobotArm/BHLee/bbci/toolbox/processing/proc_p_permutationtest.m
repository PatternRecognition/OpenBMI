function p = proc_p_permutationtest(fv,varargin)
% proc_p_permutationtest - P-values for feature/label correlation using permutation test
%
% Synopsis:
%   p = proc_p_permutationtest(fv)
%   p = proc_p_permutationtest(fv, 'Property', Value, ...)
%   
% Arguments:
%  fv: Feature vector structure with fields fv.x and fv.y (class
%      targets). Training data fv.x must be a matrix of size [dim N] for
%      dim features, N samples
%   
% Returns:
%  p: [dim 1] vector of p-values for correlation between feature i and
%      class membership
%   
% Properties:
%  perms: Scalar. Number of permutations. Default: 1000
%  rank: Logical. If true, the ranks of features values are used rather
%      than the original values. Default: 0 (use original values)
%  verbosity: Logical. If true, output a progress indicator after each
%      permutation test. Default: 0 (be silent)
%   
% Description:
%   This routines computes correlations between features and labels. 
%   The actual correlation of fv.x and fv.y will be compared to the
%   correlation of features with a randomly permuted label. This test
%   against a random permutation will be done as often as given by the
%   'perms' option.
%   
%   Return values p(i) can be roughly interpreted as the probability of
%   getting a correlation as large as the observed value by random chance,
%   when the true correlation is zero.  If p(i) is small, say less than
%   0.05, then the correlation between class targets fv.y and feature
%   fv.x(i,:) is significant. Mind that p(i) may become larger than 1.
%   
% Examples:
%   
%   
% See also: corrcoef,randperm,ranksample
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005
% $Id: proc_p_permutationtest.m,v 1.1 2006/06/19 20:15:43 neuro_toolbox Exp $

error(nargchk(1, inf, nargin))
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'perms', 1000, ...
                        'rank', 0, ...
                        'verbosity', 0);

[dim N] = size(fv.x);

% We will later always use the data upside-down (features corresponding
% to column vectors), thus transpose here already
if opt.rank,
  % Compute ranks feature-wise
  x = zeros([N dim]);
  for i = 1:dim,
    x(:,i) = ranksample(fv.x(i,:))';
  end
else
  x = fv.x';
end
if size(fv.y,1)>2,
  error('Sorry, this only works for 2-class data');
end
% Reduce two-class coding to 0/1, this is simply the first row. Thus,
% this also works for regression targets
y = fv.y(1,:);

% Avoid using corrcoef so that we don't get too much overhead
x_std = std(x);
y_std = std(y);
% Subtract the mean from all data, to avoid doing this again in each
% computation of the correlation coefficients
x = x - ones([N 1])*mean(x,1);
y = y-mean(y);

% Covariance of the i.th feature with the label
c = (y*x)./(N-1);
% Divide by both standard deviations to get correlation coefficient
actualCorr = c./(y_std*x_std);

p_low = ones([1 dim]);
p_high = ones([1 dim]);
% With zero variance feature, we can get an enourmous amount of warnings
% "divide by zero"
w = warning;
warning off;
permCorr = zeros([dim 1]);
for p = 1:opt.perms,
  if opt.verbosity>0,
    fprintf('.');
  end
  % Compute correlation with a random permutation of the label
  r = randperm(N);
  c = (y(r)*x)./(N-1);
  permCorr = c./(y_std*x_std);
  p_low = p_low + (permCorr<=actualCorr);
  p_high = p_high + (permCorr>=actualCorr);
end
warning(w);

if opt.verbosity>0,
  fprintf('\n');
end
p = 2*min(p_low, p_high)'./(opt.perms+1);
