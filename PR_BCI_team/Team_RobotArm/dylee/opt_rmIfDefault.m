function opt= opt_rmIfDefault(opt, isdefault)
%OPT_RMIFDEFAULT - Remove default-valued fields of opt-struct
%
%Usage:
%  OPT_OUT= opt_rmIfDefault(OPT, ISDEFAULT, <NEWOPT>)
%
%Description:
%  This function removes fields of struct OPT which have the default
%  as indicated by ISDEFAULT.
%
%Input:
%  OPT         and
%  ISDEFAULT - as obtained from set_defaults
%  NEWOPT    - struct or property/value list as can be input to
%              function propertylist2struct
%
%Output: 
%  OPT_OUT - struct with updated field values.

%See also: propertylist2struct, set_defaults


Fld= fieldnames(opt);
for ii= 1:length(Fld),
  if isfield(isdefault, Fld{ii}) && getfield(isdefault, Fld{ii}),
    opt= rmfield(opt, Fld{ii});
  end
end