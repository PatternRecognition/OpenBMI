function C = union2(A,B)
% UNION2 - Set union for integer sets and logical vectors
%
%   UNION2(A,B) gives the same result as UNION(A,B), yet A and B
%   must be sets of positive integers.
%   UNION2 is much faster than UNION.
%
%   UNION2 can also be called with one or both element being a logical array
%   (the characteristic vector of the sets), with 1's indicating elements
%   that are in the set. In this case, the result is also a logical array.
%
%   See also UNION
%

% 
% Copyright (c) by Anton Schwaighofer (2002)
% $Revision: 1.1 $ $Date: 2004/08/18 09:10:17 $
% mailto:anton.schwaighofer@gmx.net
% 

error(nargchk(2, 2, nargin));

if islogical(A) & islogical(B),
  if ~all(size(A)==size(B)),
    error('For logical arguments A and B, the sizes of A and B must match');
  end
  C = A | B;
elseif islogical(A),
  C = A;
  C(B) = 1;
elseif islogical(B),
  C = B;
  C(A) = 1;
else
  A = A(:)';
  B = B(:)';
  if isempty(A)
    ma = 0;
  else
    ma = max(A);
  end
  
  if isempty(B)
    mb = 0;
  else
    mb = max(B);
  end
  
  if ma==0 & mb==0
    C = [];
  elseif ma==0 & mb>0
    C = B;
  elseif ma>0 & mb==0
    C = A;
  else
    bits = false([1 max(ma,mb)]);
    bits(A) = 1;
    bits(B) = 1;
    C = find(bits);
  end
end
