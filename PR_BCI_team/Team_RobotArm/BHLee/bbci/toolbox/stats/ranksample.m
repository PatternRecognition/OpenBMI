function [r, tieadj] = ranksample(x, varargin)
% ranksample - Compute the rank of a sample, averaging or randomizing ties 
%
% Synopsis:
%   r = ranksample(x)
%   [r,tieadj] = ranksample(x, 'Property', Value, ...)
%   
% Arguments:
%  x: [m n] matrix. The data to compute the rank for.
%   
% Properties:
%  ties: String. One of 'average', 'randperm'. For tied values, either
%    return their average rank, or assign a random permutation to the
%    tied values.
%
% Returns:
%  r: [m n] matrix, the rank for data element x(i,j) is given in r(i,j)
%  tieadj: Scalar. Adjustment value for ties required by Wilcoxon signed
%      rank test and rank sum test. Only use this with 'ties'=='average'.
%   
% Description:
%   ranksample computes the rank of the values in matrix or vector x. 
%   For matrix inputs, all matrix elements are "flattened out",
%   effectively processing x(:)
%   With the 'ties'=='average' option: If any values of x are tied, the
%     returned value is their average rank.
%   With the 'ties'=='randperm' option: Ties are broken randomly, i.e. by
%     assigning a random permutation to the tied values.
%   
% Examples:
%   ranksample([1 0; 3 2]) returns
%       2     1
%       4     3
%   ranksample([1 0; 2 2]) returns
%            2            1
%          3.5          3.5
%   (The 2's would be ranked 3 and 4, their average is thus 3.5)
%
%   ranksample([1 1 1 1], 'ties', 'randperm') will return a random
%   permutation of the ranks 1:4.
%   
% See also: sort
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005
% $Id: ranksample.m,v 1.1 2005/10/07 09:34:09 neuro_toolbox Exp $

error(nargchk(1, inf, nargin));
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'ties', 'average');

N = prod(size(x));
% Sort the flattened matrix
[sx, ind] = sort(x(:));
ranks = 1:N;

% Adjust for ties
tieloc = find(diff(sx)==0);
tieadj = 0;
while length(tieloc) > 0,
  tiestart = tieloc(1);
  ntied = 1 + sum(sx(tiestart) == sx(tiestart+1:end));
  tieadj = tieadj + ntied*(ntied-1)*(ntied+1)/2;
  switch opt.ties
    case 'average'
      brokenties = tiestart + (ntied-1)/2;
    case 'randperm'
      brokenties = tiestart + randperm(ntied)-1;
    otherwise
      error('Invalid value for option ''ties''');
  end
  ranks(tiestart:tiestart+ntied-1) = brokenties;
  tieloc = tieloc(ntied:end);
end
r = zeros(size(x));
r(ind) = ranks;
