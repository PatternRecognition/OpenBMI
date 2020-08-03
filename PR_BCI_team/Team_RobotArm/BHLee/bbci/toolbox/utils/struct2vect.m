function v = struct2vect(s,fields)
% struct2vect - Pack parts of structure array into a vector
%
% Synopsis:
%   v = struct2vect(s,fields)
%   
% Arguments:
%   s: Structure array of size [1 1]
%   fields: [1 f] cell array, describes which parts of s to write to the vector
%       v. Entry {'fname', ind} means to copy s.fname(ind) into v. Entry {'fname'}
%       or 'fname' will copy s.fname into v.
%   
% Returns:
%   v: [1 f] vector, contains the elements of s selected by fields.
%   
% Description:
%   This helper routine can be used to pack parts of a structure array
%   into a vector. This is useful, for example, to pack options into a
%   vector that is passed on to a numeric optimizer.
%   Use vect2struct to perform the inverse operation.
%   
% Examples:
%   s = struct('f1', [1 2], 'f2', 3);
%   struct2vect(s, {{'f1', 2}, 'f2'})
%     will return the vector [2 3]
%   
% See also: vect2struct
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: struct2vect.m,v 1.1 2005/04/07 15:10:52 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));
if ~iscell(fields),
  error('Invalid input ''fields''');
end
if ~isstruct(s) | prod(size(s))~=1,
  error('Input ''s'' must be a struct array of size [1 1]');
end
fields = {fields{:}};
v = NaN*zeros([1 length(fields)]);
for i = 1:length(fields),
  p = fields{i};
  if iscell(p),
    if length(p)==1 & ischar(p{1}),
      % Entry is of the form {'fieldname'}
      f = getfield(s, p{1});
      if prod(size(f))~=1,
        error('Unable to copy array into vector');
      end
      v(i) = f;
    elseif length(p)==2 & ischar(p{1}) & isnumeric(p{2}),
      % Entry is of the form {'fieldname', index}. getfield can access
      % the index directly, need to pass the index as *cell array*
      v(i) = getfield(s, p{1}, p(2));
    else
      error(sprintf('Invalid entry in fields{%i}', i));
    end
  elseif ischar(p),
    % Entry is of the form 'fieldname'
    f = getfield(s, p);
    if prod(size(f))~=1,
      error('Unable to copy array into vector');
    end
    v(i) = getfield(s, p);
  else
    error(sprintf('Invalid entry in fields{%i}', i));
  end
end
