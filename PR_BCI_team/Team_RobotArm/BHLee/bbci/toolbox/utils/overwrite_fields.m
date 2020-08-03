function S= overwrite_fields(S, varargin);
%OVERWRITE_FIELDS - Overwrite fields of a struct
%
%Synopsis:
% Sout= overwrite_fields(Sin, 'FLD1',VAL1, 'FLD2','VAL2', ...)
% Sout= overwrite_fields(Sin, {'FLD1',VAL1, 'FLD2','VAL2', ...})
%
%Comment:
% Can also be used to set fields of a substructure,
%   overwrite_fields([], 'a.b.c', [17 24]);
% Therefore the implementation is done with 'eval'.

if length(varargin)==1,
  C= varargin{1};
else
  C= varargin;
end
if isstruct(C),
  C= struct2propertylist(C);
end

for ii= 1:length(C)/2,
  cmd= sprintf('S.%s= C{2*ii};', C{2*ii-1});
  eval(cmd);  % see comment in the help
end
