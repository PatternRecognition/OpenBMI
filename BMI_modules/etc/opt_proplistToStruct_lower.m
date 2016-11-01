function opt = opt_proplistToStruct(varargin)
% OPT_PROPLISTTOSTRUCT - Make options struct from parameter/value list
%
%Synopsis:
%  OPT= opt_proplistToStruct(<'Param1', VALUE1, 'Param2', VALUE2, ...>)
%  OPT= opt_proplistToStruct(OPT_IN, <'Param1', VALUE1, 'Param2', VALUE2, ...>)
%
%Arguments:
%  OPT_IN:      STRUCT of optional properties
%  'Param1', VALUE1, ...: Property/value list
%
%Returns:
%  OPT:  STRUCT with (new) fields created from the property/value list
%
%Description:
%  Generates a struct OPT (or updates a given struct OPT_IN) with field
%  'Param1' set to value VALUE1, field 'Param2' set to value VALUE2, and
%  so forth (property/value list).
%
%See also opt_setDefaults opt_structToProplist

% 06-2012 Benjamin Blankertz


opt= [];
if nargin==0,
  return;
end

if isstruct(varargin{1}),
  % First input argument is already a structure: Start with that, write
  % the additional fields
  opt= varargin{1};
  iListOffset= 1;
elseif isempty(varargin{1}),
  % First input argument is empty, which can happen under some
  % circumstances. This is not an opt, but also not part of the list
  iListOffset = 1;
else
  % First argument is not a structure: Assume this is the start of the
  % parameter/value list
  iListOffset = 0;
end

nFields= (nargin-iListOffset)/2;
if nFields~=round(nFields),
  error('Invalid parameter/value list');
end

for ff= 1:nFields,
  fld = lower(varargin{iListOffset+2*ff-1});
  if ~ischar(fld),
    error('Invalid parameter/value list');
  end
  opt.(fld)= lower(varargin{iListOffset+2*ff});
end
