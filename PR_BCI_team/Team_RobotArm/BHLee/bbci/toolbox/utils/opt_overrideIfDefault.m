function [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, varargin)
%OPT_OVERRIDEIFDEFAULT - Override default-valued fields of opt-struct
%
%Usage:
%  OPT_OUT= opt_overrideIfDefault(OPT, ISDEFAULT, <NEWOPT>)
%
%Description:
%  This function updates fields of struct OPT with values of
%  struct (or property/value list) NEWOPT, but only those fields that
%  had obtained values by default (in the function set_defaults),
%  as recorded by ISDEFAULT (and fields that do not exist in OPT).
%
%Input:
%  OPT         and
%  ISDEFAULT - as obtained from set_defaults
%  NEWOPT    - struct or property/value list as can be input to
%              function propertylist2struct
%
%Output: 
%  OPT_OUT - struct with updated field values.
%  ISDEFAULT - struct with updated information, i.e., isdefault is set
%              to false for all given fields.
%
%Example:
%  opt= struct('type', 'large');
%  [opt, isdefault]= set_defaults(opt, ...
%    'lineWidth', 1, ...
%    'fontSize', 10, ...
%    'type', 'normal');
%  if strcmpi(opt.type, 'large'),
%    [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
%        'lineWidth', 3, ...
%        'fontSize', 18);
%  end
%
%See also: propertylist2struct, set_defaults

% Author(s): Benjamin Blankertz, Jan 2005


%if ~isequal(fieldnames(opt), fieldnames(isdefault)),
%  warning('opt and isdefault (i.e. 1st and 2nd argument) have different fields');
%end

newopt= propertylist2struct(varargin{:});
Fld= fieldnames(newopt);
for ii= 1:length(Fld),
  if ~isfield(opt, Fld{ii}) | getfield(isdefault, Fld{ii}),
    opt= setfield(opt, Fld{ii}, getfield(newopt, Fld{ii}));
    isdefault= setfield(isdefault, Fld{ii}, 0);
  end
end
