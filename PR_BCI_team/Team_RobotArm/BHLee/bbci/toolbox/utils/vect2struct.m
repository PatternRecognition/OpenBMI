function s = vect2struct(s,v,fields)
% vect2struct - Copy a vector into parts of a struct array
%
% Synopsis:
%   s = vect2struct(s,v,fields)
%   
% Arguments:
%   s: Structure array of size [1 1]
%   v: [1 f] vector, containing the data to be copied into struct s.
%   fields: [1 f] cell array, describes the position of where to write the
%       elements of v. fields{i} gives the position for v(i). Entries can be
%       {'fname', ind}, to write v(i) to s.fname(ind). Entry {'fname'} or 'fname'
%       writes v(i) to s.fname.
%   
% Returns:
%   s: Updated struct array, a copy of input argument s with the chosen fields
%       overwritten by values in v.
%   
% Description:
%   This routine is the 'inverse' of struct2vect. It copies a vector into
%   specified positions of a structure array. A typical use would be to
%   take the results of a numeric optimizer (model parameters), and
%   convert them back to an options structure format.
%   
%   
% Examples:
%   s = struct('f1', [1 2], 'f2', 3);
%   vect2struct(s, [9 10], {{'f1', 1}, f2})
%     will return
%     ans = 
%         f1: [9 2]
%         f2: 10
%   
% See also: struct2vect
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: vect2struct.m,v 1.1 2005/04/07 15:10:52 neuro_toolbox Exp $

error(nargchk(3, 3, nargin));
if ~iscell(fields),
  error('Invalid input ''fields''');
end
if ~isstruct(s) | prod(size(s))~=1,
  error('Input ''s'' must be a struct array of size [1 1]');
end
fields = {fields{:}};
v = v(:);
if length(fields)~=length(v),
  errror('Inputs ''v'' and ''fields'' must have matching length');
end
for i = 1:length(v),
  p = fields{i};
  if iscell(p),
    if length(p)==1 & ischar(p{1}),
      % Entry is of the form {'fieldname'}
      if prod(size(getfield(s, p{1})))~=1,
        error('Unable to write scalar into array');
      end
      s = setfield(s, p{1}, v(i));
    elseif length(p)==2 & ischar(p{1}) & isnumeric(p{2}),
      % Entry is of the form {'fieldname', index}. setfield can access
      % the index directly, need to pass the index as *cell array*
      s = setfield(s, p{1}, p(2), v(i));
    else
      error(sprintf('Invalid entry in fields{%i}', i));
    end
  elseif ischar(p),
    % Entry is of the form 'fieldname'
    if prod(size(getfield(s, p)))~=1,
      error('Unable to write scalar into array');
    end
    s = setfield(s, p, v(i));
  else
    error(sprintf('Invalid entry in fields{%i}', i));
  end
end
