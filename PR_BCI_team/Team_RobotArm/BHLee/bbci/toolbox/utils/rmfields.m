function S= rmfields(S, varargin)
%S= rmfields(S, flds)
%
% rmfields removes one or more fields from a struct.
%
% S is a struct and flds is a cell array or a list of field names.

if length(varargin)==1,
  if iscell(varargin{1}),
    flds= varargin{1};
  else
    flds= {varargin{1}};
  end
else
  flds= varargin;
end

for ff= 1:length(flds),
  if isfield(S, flds{ff}),
    S= rmfield(S, flds{ff});
  else
    msg= sprintf('warning %s not in struct', flds{ff});
    warning(msg);
  end
end
