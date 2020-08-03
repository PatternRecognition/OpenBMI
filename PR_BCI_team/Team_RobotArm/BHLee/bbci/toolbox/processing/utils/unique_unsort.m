function y= unique_unsort(x, n)
%UNIQUE - Like MATLAB's unique, but without sorting
%
%Synopsis:
%  Y= union_unsort(X, N)
%
%Description:
%  Returns the first N unique elements of X.
%
%Arguments:
%  X - can be a vector or a cell array of strings
%
%% TODO: should be written more generally for matrices

if nargin<2,
  n= inf;
end

y= x([]);  %% empty matrix or empty cell
ii= 0;
while ii<length(x) & length(y)<n,
  ii= ii + 1;
  if ~ismember(x(ii), y),
    y= [y, x(ii)];
  end
end
