function opt = propertylist2struct(varargin)
% PROPERTYLIST2STRUCT - Make options structure from parameter/value list
%
%   OPT= propertylist2struct('param1', VALUE1, 'param2', VALUE2, ...)
%   Generate a structure OPT with fields 'param1' set to value VALUE1, field
%   'param2' set to value VALUE2, and so forth.
%
%   See also set_defaults

opt= [];
if nargin==0,
  return;
end

if isstruct(varargin{1}) | isempty(varargin{1}),
  % First input argument is already a structure: Start with that, write
  % the additional fields
  opt= varargin{1};
  iListOffset= 1;
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
  fld = varargin{iListOffset+2*ff-1};
  if ~ischar(fld),
    error('Invalid parameter/value list');
  end
%  prp= varargin{iListOffset+2*ff};
%  opt= setfield(opt, fld, prp);
  opt.(fld)= varargin{iListOffset+2*ff};
end
