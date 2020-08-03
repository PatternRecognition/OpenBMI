function fv_bin = proc_binomialtest(fv,varargin)
% fv_wil = proc_binomialtest(fv,<p0,alpha>)
%
% binomial test value for the first feature dimension.
% the first dim is assumed to have values 0 and 1.
% If fv_bin.x is not inside the interval defined by 
% fv_bin.crit, then H_0:(p==p0) can be rejected.
%
% IN     fv   - data structure of feature vectors
%        alpha- significance level for hypothesis H_0.
%        p0   - assumed parameter to test for.
% OUT    fv_bin- data structure of wilcoxon values (one sample only)
%            .x - normalized binomial test statistics
%            .crit- critical value for significance level alpha.
% SEE    proc_r_values

% kraulem 11/06
switch length(varargin)
 case 0
  alpha = .01;
  p0 = .5;
 case 1
  alpha = .01;
  p0 = varargin{1};
 case 2
  alpha = varargin{2};
  p0 = varargin{1};
end

% check input arguments
if ~ismember(1,(fv.x))|ndims(fv.x)>2
  error('fv must be a vector!');
end
if ~isempty(setdiff(unique(fv.x),[0 1]))
  error('fv must have only 0s and 1s');
end

bin = sum(fv.x);
n = length(fv.x);

bin = (bin - n*p0)/sqrt(n*p0*(1-p0));

% construct a new fv:
fv_bin= copyStruct(fv, 'x','y','className');
fv_bin.x= bin;
fv_bin.y= 1;
fv_bin.yUnit= 'bin';
fv_bin.crit = normal_invcdf(1-alpha/2);
fv_bin.crit = [-fv_bin.crit fv_bin.crit];
return