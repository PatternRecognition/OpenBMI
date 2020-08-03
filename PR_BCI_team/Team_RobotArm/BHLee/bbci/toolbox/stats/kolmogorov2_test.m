function [H,pValue,KSstatistic] = kolmogorov2_test(x1 , x2 , varargin)
% kolmogorov2_test - Two-sample Kolmogorov-Smirnov goodness-of-fit hypothesis test
%
% Synopsis:
%   H = kolmogorov2_test(x1,x2)
%   [H,pValue,KSstatistic] = kolmogorov2_test(x1,x2)
%   [H,pValue,KSstatistic] = kolmogorov2_test(x1,x2,'Property',Value,...)
%   
% Arguments:
%  x1: [1 n1] vector, first sample of data. If matrix is given, all
%      matrix entries are treated as the data sample. Missing
%      observations, indicated by NaN's, are ignored.
%  x2: [1 n2] vector, second sample of data.
%   
% Returns:
%  H: Scalar. result of the hypothesis test:
%     H = 0: Do not reject the null hypothesis at significance level <alpha>
%     H = 1: Reject the null hypothesis at significance level <alpha>
%  pValue: Scalar, asymptotic P-value
%  KSstatistic: Scalar, the Kolmogorov-Smirnof test statistic for the
%      test defined by option <tail>
%   
% Properties:
%  alpha: Scalar, desired significance level. Default: 0.05
%  tail: One of [0, 1, -1]. Defines the type of test that is used
%     The null hypothesis is always F1(x) = F2(x) for all x. Alternative
%     hypothesis H1 depends on <tail>:
%     <tail>= 0 (2-sided test) H1: F1(x) not equal to F2(x).
%     <tail>= 1 (1-sided test) H1: F1(x) > F2(x).
%     <tail>=-1 (1-sided test) H1: F1(x) < F2(x).
%
% Description:
%   kolmogorov2_test(x1,x2) performs a Kolmogorov-Smirnov (K-S) test to
%   determine if independent random samples, x1 and x2, are drawn from the
%   same underlying continuous population.
% 
%   Let F1(x) and F2(x) be the empirical distribution functions from sample 
%   vectors x1 and x2, respectively. The 2-sample K-S test hypotheses and 
%   test statistic are:
%     The null hypothesis is always F1(x) = F2(x) for all x. Alternative
%     hypothesis H1 depends on <tail>:
%     <tail>= 0 (2-sided test) H1: F1(x) not equal to F2(x). (Default value)
%     <tail>= 1 (1-sided test) H1: F1(x) > F2(x).
%     <tail>=-1 (1-sided test) H1: F1(x) < F2(x).
%
%   The test statistics are T = max|F1(x) - F2(x)| (for <tail> = 0)
%   T = max[F1(x) - F2(x)] (for <tail>=1), and T = max[F2(x) - F1(x)]
%   (for <tail> = -1)
%
%   The decision to reject the null hypothesis occurs when the significance 
%   level, <alpha>, equals or exceeds the P-value.
%
%   The asymptotic P-value becomes very accurate for large sample sizes, and
%   is believed to be reasonably accurate for sample sizes N1 and N2 such 
%   that (N1*N2)/(N1 + N2) >= 4.
%   
% Examples:
%   Compare 25 samples from normal and uniform distribution:
%     kolmogorov2_test(rand(5), randn(5))
%       ans =
%            0
%   We can not (yet) reject the hypothesis that the underlying
%   distributions are equal
%   After increasing the number of samples to 100, we can:
%     kolmogorov2_test(rand(10), randn(10))
%       ans =
%            1
%   
% References:
%   (1) Massey, F.J., "The Kolmogorov-Smirnov Test for Goodness of Fit",
%         Journal of the American Statistical Association, 46 (March 1956), 68-77.
%   (2) Miller, L.H., "Table of Percentage Points of Kolmogorov Statistics",
%         Journal of the American Statistical Association, (March 1951), 111-121.
%   (3) Conover, W.J., "Practical Nonparametric Statistics", 
%         John Wiley & Sons, Inc., 1980.
%
% See also: 
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2006
% Based on code from the stats toolbox
% $Id: kolmogorov2_test.m,v 1.1 2006/06/19 20:34:59 neuro_toolbox Exp $

error(nargchk(2, inf, nargin));
opt = propertylist2struct(varargin{:});
[opt, isdefault] = set_defaults(opt, 'alpha', 0.05, ...
                                     'tail', 0);

% Always treat all data as the sample
x1 = x1(:);
x2 = x2(:);

% Always remove missing observations indicated by NaN's
x1 = x1(~isnan(x1));
x2 = x2(~isnan(x2));

if isempty(x1)
  error('Empty observation matrix X1');
end
if isempty(x2)
  error('Empty observation matrix X2');
end


% Ensure the significance level, ALPHA, is a scalar between 0 and 1
if prod(size(opt.alpha)) > 1 | opt.alpha <= 0 | opt.alpha >= 1,
  error('Significance level <alpha> must be a scalar between 0 and 1.');
end

% Ensure the type-of-test indicator, TAIL, is a scalar integer from 
% the allowable set {-1 , 0 , 1}
if prod(size(opt.tail)) > 1 | ~isempty(setdiff(opt.tail, [0 -1 1])),
  error('Type-of-test indicator <tail> must be a one of {-1, 0, 1}.');
end

% Calculate F1(x) and F2(x), the empirical (i.e., sample) CDFs.
binEdges    =  [-inf ; sort([x1;x2]) ; inf];

binCounts1  =  histc (x1 , binEdges);
binCounts2  =  histc (x2 , binEdges);

sumCounts1  =  cumsum(binCounts1)./sum(binCounts1);
sumCounts2  =  cumsum(binCounts2)./sum(binCounts2);

sampleCDF1  =  sumCounts1(1:end-1);
sampleCDF2  =  sumCounts2(1:end-1);

% Compute the test statistic of interest.
switch opt.tail
  case  0      
    %  2-sided test: T = max|F1(x) - F2(x)|.
    deltaCDF  =  abs(sampleCDF1 - sampleCDF2);
  case -1      
    %  1-sided test: T = max[F2(x) - F1(x)].
    deltaCDF  =  sampleCDF2 - sampleCDF1;
  case  1
    %  1-sided test: T = max[F1(x) - F2(x)].
    deltaCDF  =  sampleCDF1 - sampleCDF2;
end

KSstatistic   =  max(deltaCDF);

% Compute the asymptotic P-value approximation and accept or
% reject the null hypothesis on the basis of the P-value.

n1     =  length(x1);
n2     =  length(x2);
n      =  n1 * n2 /(n1 + n2);
lambda =  max((sqrt(n) + 0.12 + 0.11/sqrt(n)) * KSstatistic , 0);

if opt.tail ~= 0        
  % 1-sided test.
  pValue  =  exp(-2 * lambda * lambda);
else                
  % 2-sided test (default):
  %  Use the asymptotic Q-function to approximate the 2-sided P-value.
  j       =  [1:101]';
  pValue  =  2 * sum((-1).^(j-1).*exp(-2*lambda*lambda*j.^2));
  pValue = max(0, pValue);
  pValue = min(1, pValue);
end

H = (opt.alpha >= pValue);
