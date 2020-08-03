function T= struct_copyFields(T, S, fld_list)
%STRUCT_COPYFIELDS - Copy specified fields from one struct (into another one)
%
%Synopsis:
%  T= struct_copyFields(S, <FLD_LIST>)
%  T= struct_copyFields(T, S, <FLD_LIST>)
%
%Arguments:
%  S:        STRUCT       - Source, from which fields are taken
%  T:        STRUCT       - Target, into which fields are pasted
%  FLD_LIST: CELL of CHAR - Names of the fields that should be copied from
%      the 'source' struct S (into a 'target' struct T, if provided).
%      If a fieldname has prefix '!', it is required to exist (otherwise
%      an error is thrown).
%      If a fieldname has prefix '~', this function tolerates nonexistence
%      of that field silently.
%      Otherwise, a warning is issued for nonexisting fields.
%
%Returns:
%  T:  STRUCT with specified fields from S inserted


misc_checkType(T, 'STRUCT');

if nargin<3 && ~isstruct(S),
  % T= struct_copyFields(S, <FLD_LIST>)
  T= struct_copyFields([], T, S);
  return;
end

misc_checkType(S, 'STRUCT');
misc_checkTypeIfExists('fld_list', 'CHAR|CELL{CHAR}');


if nargin<3,
  fld_list= fieldnames(S);
end
% Let us be gracious:
if ischar(fld_list),
  fld_list= {fld_list};
end

warning_list= {};
for ii= 1:length(fld_list),
  fld= fld_list{ii};
  require_existence= 0;
  tolerate_nonexistence= 0;
  if fld(1)=='!',
    require_existence= 1;
    fld(1)= [];
  elseif fld(1)=='~',
    tolerate_nonexistence= 1;
    fld(1)= [];
  end
  if ~isfield(S, fld),
    if require_existence,
      error(sprintf('Field ''%s'' not found in struct', fld));
    end
    if ~tolerate_nonexistence,
      warning_list= cat(2, warning_list, {fld});
    end
    continue;
  end
  T.(fld)= S.(fld);
end

if ~isempty(warning_list),
  warning(sprintf('Field(s) not found in struct: %s', vec2str(warning_list)));
end
