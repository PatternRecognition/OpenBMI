function i = str2ind(S,ind)
% str2ind - Indexing with string sets and general sets
%
% Synopsis:
%   i = str2ind(S,ind)
%   
% Arguments:
%   S: Cell array of strings
%   ind: Cell array of strings to compute the index for
%
%   S and ind can be of other types as well, valid types are those accepted
%   in calling ismember(ind, S).
%   
% Returns:
%   i: Indices in S such that S(i)==ind
%   
% Description:
%   This is a simple wrapper function for Matlab's ismember
%   function. str2ind returns the second output argument of ismember(ind,S).
%   Also, the function raises an error if any member of ind is not
%   contained in S.
%   
%   
% Examples:
%   S = {'The', 'quick', 'brown', 'fox'};
%   str2ind(S, {'fox', 'The'})
%   ans = 
%        4    1
%   str2ind(S, 'fox')
%   ans = 
%        4
%   str2ind(S, 'Brown') 
%     raises an error.
%   str2ind([17 18 19 20], 18)
%   ans =
%        2
%   
% See also: ismember
% 

% Author(s): Anton Schwaighofer, Dec 2004
% $Id: str2ind.m,v 1.1 2005/01/31 09:36:41 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

[dummy, i] = ismember(ind, S);
if any(i==0),
  error('Invalid indexing set. Argument ''S'' must be a superset of ''ind''');
end
