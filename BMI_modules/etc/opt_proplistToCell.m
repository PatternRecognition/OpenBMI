function [ opt ] = opt_proplistToCell( varargin )
%OPT_PROPLISTTOCELL Summary of this function goes here
%   Detailed explanation goes here

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
  opt{ff,1}=varargin{iListOffset+2*ff-1};
  opt{ff,2}=varargin{iListOffset+2*ff};
end


end

