function siglevel = paired_ttest(A,B,dim)
% paired_ttest - Paired T-test to compare sample differences
%
% Synopsis:
%   siglevel = paired_ttest(A,B)
%   siglevel = paired_ttest(A,B,dim)
%   
% Arguments:
%  A: [n m] matrix, samples A
%  B: [n m] matrix, samples B
%  dim: Scalar. Dimension along which to operate. Default: 1
%   
% Returns:
%  siglevel: Scalar. Significance level of the hypothesis "mean(A) is
%      greater than mean(B)". If the result siglevel<0, abs(siglevel) is the
%      significance level of the hypothesis "mean(B) is greater than mean(A)".
%   
% Description:
%   This implements the paired t-test to compare the mean of two
%   distributions. For example, A and B can be the mean squared errors of
%   two classifiers A, B on 10 data sets. 
%
%   With given dimension dim, the test operates along the given dimension,
%   that is, paired_ttest compares mean(B,dim) with mean(A,dim). siglevel is
%   in this case a vector or matrix with significance levels computed as
%   described above.
%   
% Examples:
%   paired_ttest([1 1 2 2], [3 3 4 5], 2)
%     return -0.9985, showing that the mean of the second argument is
%     significantly larger than the mean of the first argument
%   
% See also: t_cdf
% 

% Author(s), Copyright: Anton Schwaighofer, Aug 2005
% $Id: paired_ttest.m,v 1.1 2005/09/02 15:07:56 neuro_toolbox Exp $

error(nargchk(2, 3, nargin));
if nargin<3,
  dim = 1;
end

if ~all(size(A)==size(B)),
  error('Inputs A and B must have matching size');
end
repsize = ones([1 ndims(A)]);
repsize(dim) = size(A,dim);
ma = mean(A, dim);
mb = mean(B, dim);
% Degrees of freedom
dof = size(A,dim)-1;
if dof<1,
  error(sprintf('There is only one sample available along dimension %i', ...
                dim));
end
% Value for t distribution
t = (ma-mb).*sqrt((dof*(dof+1))./sum((A-repmat(ma,repsize)-B+repmat(mb,repsize)).^2,dim));
% Significance level
siglevel = sign(t).*t_cdf(abs(t), dof);
