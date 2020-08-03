function [C,range] = countTuples(data, varargin)
% countTuples - Count tuples of value co-occurences in matrix
%
% Synopsis:
%   C = countTuples(data)
%   [C,range] = countTuples(data,'Property',Value,...)
%   
% Arguments:
%  data: [dim N] matrix, representing N tuples of length dim each.
%   
% Returns:
%  C: N-D array with dim dimensions. C(i,j,...k) is the number of
%      occurences of the tuple (range1(i), range2(j), .... rangedim(k))
%  range: [1 dim] cell array. range{i} is the values taken on
%      by the i.th row of the data matrix
%   
% Properties:
%  range: [1 dim] cell array, assumed range for the i.th data column
%   
% Description:
%   Count how many tuples of a certain kind appear in a data matrix,
%   resp. count how many different columns there are in the matrix.
%   This can be used on data of arbitrary dimensions (tuples of arbitrary
%   dimension)
%   
% Examples:
%   [C,range]=countTuples([1 2;3 4;1 2;1 4]')
%      C =
%           2     1
%           0     1
%      range = 
%          {[1 3]  [2 4]}
%   C(1,2)==1 means that the tuple (range{1}(1) range{2}(2)) occurs once.
%   Option <range> can be used to specify a predefined range for rows:
%   countTuples([1 2;3 4;1 2]', 'range', {[1 2 3], [2 3 4]})
%     ans =
%          2     0     0
%          0     0     0
%          0     0     1
%
% See also: unique
% 

% Author(s), Copyright: Anton Schwaighofer, Feb 2006
% $Id: countTuples.m,v 1.2 2006/02/08 11:47:06 neuro_toolbox Exp $

error(nargchk(1, inf,nargin))
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'range', []);

[dim N] = size(data);

if isempty(opt.range),
  % Extract the data range for each column of data
  for i = 1:dim,
    range{i} = unique(data(i,:));
  end
else
  range = opt.range;
  for i = 1:dim,
    if ~isempty(setdiff(unique(data(i,:)), range{i})),
      error('Option <range> must contain a superset of values found in data(:,i)');
    end
  end  
end
for i = 1:dim,
  l(i) = length(range{i});
end
% C contains an entry for each possible co-occurence
C = zeros(l);

% Extract all occuring tuples, plus the index that says how many
% occurences there are
[tuples,dummy,count] = unique(data', 'rows');
for j = 1:size(tuples,1),
  % Check all tuples, and see where we should put that in the C matrix:
  tj = tuples(j,:);
  ind = cell([1 dim]);
  for i = 1:dim,
    ind{i} = find(tj(i)==range{i});
  end
  C(sub2ind(size(C), ind{:})) = nnz(count==j);
end
