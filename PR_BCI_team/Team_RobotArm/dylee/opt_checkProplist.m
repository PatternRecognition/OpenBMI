function opt_checkProplist(opt, props, varargin)
%OPT_CHECKPROPLIST - Check a property/value struct according to specifications
%
%Synopsis:
%  opt_checkProplist(OPT, PROPSPEC, <PROPSPEC2, ...>)
%
%Arguments:
%  OPT:      STRUCT of optional properties
%  PROPSPEC: PROPSPECLIST - Property specification list, i.e., CELL of size
%      [N 2] or [N 3], with the first column all being strings.
%
%Returns:
%  nothing (just throws errors)
%
%Description:
%  This function checks whether all fields of OPT occur in the list of
%  property names of PROPSPEC and throws an error otherwise. This check
%  is case-insensitive.
%  If PROPSPEC contains a third column of (type definitions), the values
%  in OPT are checked to match them, see opt_checkTypes.
%
%  Type checking can be switched off (and on again) by the function
%  bbci_typechecking which is useful for time critical operations.
%
%See also opt_checkTypes, bbci_typechecking.
%
%Examples:
%  props= {'LineWidth', 2, 'DOUBLE[1]'; 'Color', 'k', 'CHAR|DOUBLE[3]'}
%  opt= struct('Color','g', 'LineStyle','-');
%  % This should throw an error:
%  opt_checkProplist(opt, props)
%  opt= struct('Color',[0.5 0.2 0.3 0.1]);
%  % This should also throw an error:
%  opt_checkProplist(opt, props)
%  opt= struct('Color',[0.1 0.6 0.3]);
%  % This is ok:
%  opt_checkProplist(opt, props)

% 06-2012 Benjamin Blankertz


global BTB

% if ~BTB.TypeChecking, return; end

if length(varargin)>0 && ischar(varargin{end}),
  structname= varargin{end};
  varargin(end)= [];
else
  structname= '';
end

props_all= cat(1, props, varargin{:});
fn= fieldnames(opt);
isknown= ismember(upper(fn), upper(props_all(:,1)),'legacy');
unknown_fields= fn(~isknown);
if ~isempty(unknown_fields),
  if length(unknown_fields)==1,
    tag= 'unexpected property';
  else
    tag= 'unexpected properties';
  end
  if ~isempty(structname),
    tag= [tag sprintf(' in STRUCT ''%s''', structname)];
  end
  error('%s: %s.', tag, str_vec2str(unknown_fields));
end

opt_checkTypes(opt, props, structname);