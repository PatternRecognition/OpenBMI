function [opt, isdefault]= set_defaults(opt, varargin)
%[opt, isdefault]= set_defaults(opt, field/value list)
%
%Description:
% This functions fills in the given struct opt some new fields with
% default values, but only when these fields DO NOT exist before in opt.
% Existing fields are kept with their original values.
%
%Example:
%   opt= set_defaults(opt, 'color','g', 'linewidth',3);
%
% The second output argument isdefault is a struct with the same fields
% as the returned opt, where each field has a boolean value indicating
% whether or not the default value was inserted in opt for that field.

% blanker@cs.tu-berlin.de

% Set 'isdefault' to ones for the field already present in 'opt'
isdefault= [];
if ~isempty(opt),
  for Fld=fieldnames(opt)',
    isdefault.(Fld{1})= 0;
  end
end

defopt = propertylist2struct(varargin{:});
for Fld= fieldnames(defopt)',
  fld= Fld{1};
  if ~isfield(opt, fld),
    %% if opt is a struct *array*, the fields of all elements need to
    %% be set. This is done with the 'deal' function.
    [opt.(fld)]= deal(defopt.(fld));
    isdefault.(fld)= 1;
  end
end
