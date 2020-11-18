function [opt, isdefault]= opt_setDefaults(opt, props, check)
%OPT_SETDEFAULTS - Set default values according to a property spec list
%
%Synopsis:
%  [OPT, ISDEFAULT]= opt_setDefaults(OPT, PROPSPEC)
%  [OPT, ISDEFAULT]= opt_setDefaults(OPT, PROPSPEC, CHECK)
%
%Arguments:
%  OPT:      STRUCT of optional properties
%  PROPSPEC: PROPSPECLIST - Property specification list, i.e., CELL of size
%      [N 2] or [N 3], with the first column all being strings.
%  CHECK:    BOOL - Evokes type checking according to the third column of
%      PROPSPEC
%
%Returns:
%  OPT: STRUCT with added properties from the property specification list
%          PROPSPEC
%  ISDEFAULT: STRUCT with the same fields as OPT. Each field has a
%          boolean value indicating whether the default value from PROPSPEC
%          was taken (i.e., that property was not a field of the input OPT).
%
%Description:
%  PROPSPEC is a property specification list that defines default values
%  for a list of properties. The struct OPT is filled with all missing
%  properties from PROPSPEC, but existing fields of OPT are not overwritten
%  by the respective default values.
%  The function makes a case-insensitive match of field of OPT to the
%  property names in PROPLIST. In the case of a match, the variant of
%  PROPSPEC is used.
%
%See also opt_checkProplist, opt_proplistToStruct.
%
%Examples:
%  opt= struct('linewidth',3, 'color','k')
%  props= {'LineWidth', 2;  'Color',[0 0 1];  'LineStyle','--'}
%  [opt, isdefault]= opt_setDefaults(opt, props)
%
%  % If the first input variable (OPT) is not a variable itself, but a field
%  % of a struct, it has to passed as a string:
%  S.opt= opt_setDefaults('S.opt', props);

% 06-2012 Benjamin Blankertz


misc_checkType(opt, 'STRUCT|CHAR');
misc_checkType(props, 'PROPSPEC');
misc_checkTypeIfExists('check', 'BOOL');

if nargin<3,
  check= 0;
end

if ischar(opt),
  structname= opt;
  opt= evalin('caller', opt);
  check= 1;
else
  structname= '';
end


% Perform a case correction in the STRUCT opt, and
% Set 'isdefault' to ones for the field already present in 'opt'
isdefault= [];
if ~isempty(opt),
  for Fld=fieldnames(opt)',
    fld= Fld{1};
    idx= find(strcmpi(fld, props(:,1)));
    if length(idx)>1,
      error('Case ambiguity in propertylist');
    elseif ~isempty(idx) && ~strcmp(fld, props{idx,1}),
      % rename field in opt to match case with props
      oldfld= fld;
      fld= props{idx,1};
      [opt.(fld)]= opt.(oldfld);
      opt= rmfield(opt, oldfld);
    end
    isdefault.(fld)= 0;
  end
end

for k= 1:size(props,1),
  fld= props{k,1};
  if ~isfield(opt, fld),
    %% if opt is a struct *array*, the fields of all elements need to
    %% be set. This is done with the 'deal' function.
    [opt.(fld)]= deal(props{k,2});
    isdefault.(fld)= 1;
  end
end

if check,
  if check==2,
    tcstate= bbci_typechecking('on');
  end
  if isempty(structname),
    structname= inputname(1);
  end
  opt_checkProplist(opt, props, structname);
  if check==2,
    bbci_typechecking(tcstate);
  end
end