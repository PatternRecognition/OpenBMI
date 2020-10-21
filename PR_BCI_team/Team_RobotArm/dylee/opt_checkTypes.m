function opt_checkTypes(opt, optDefaults, structname)
%OPT_CHECKTYPE - Check types of the values in property/value struct
%
%Synopsis:
%  opt_checkTypes(OPT, PROPSPEC)
%
%Arguments:
%  PROPSPEC: PROPSPECLIST - Property specification list with type definition,
%      i.e., CELL of size [N 3], with the first column all being strings
%      (property names), the second column holding the default values (not
%      used in this function), and the third column specifying the expected
%      type of the values. If a third column does not exist, this function
%      just exists.
%
%Returns:
%  nothing (just throws errors)
%
%Description:
%  This function ensures that the values in the property/value struct OPT
%  comply with the types that are specified in the property specification
%  PROCSPEC. For a list of possible specifications, see misc_checkType.
%
%See also misc_checkType, opt_setDefaults.
%
%Examples:
%  props= {'LineColor', 'k', 'CHAR[1]|DOUBLE[3]';
%          'LineWidth', 0.7, 'DOUBLE'; 
%          'Colormap', copper(51), 'DOUBLE[- 3]';
%          'Transform', eye(2,2), 'DOUBLE[2 2]'};
%
%  opt= struct('Colormap', hot(5));
%  opt_checkTypes(opt, props)
%
%  opt= struct('LineColor', [1 0 0.5 0.5]);
%  % Should throw an error
%  opt_checkTypes(opt, props)
%
%  opt= struct('LineWidth', 'Very Thick');
%  % Should throw an error
%  opt_checkTypes(opt, props)
%
%  opt= struct('Transform', randn(2,2));
%  opt_checkTypes(opt, props)

% 06-2012 Benjamin Blankertz


if size(optDefaults, 2) < 3,
  % No types specified
  return
end

for k= 1:size(optDefaults, 1),
  fld= optDefaults{k,1};
  if isfield(opt, fld),
    fielddisplayname= [structname '.' fld];
    for n= 1:length(opt),
      [ok, msg]= misc_checkType(opt(n).(fld), optDefaults{k,3}, fielddisplayname);
      if ~ok,
        error(msg);
      end
    end
  end
end
