function T= copy_fields(T, S, fld_list)
%COPY_FIELDS - Copy specified fields from one struct (into another one)
%
%Synopsis:
%  T= copy_fields(S, <FLD_LIST>)
%  T= copy_fields(T, S, <FLD_LIST>)
%
%Arguments:
%  S - [STRUCT] Source, from which fields are taken
%  T - [STRUCT] Target, into which fields are pasted
%  FLD_LIST - [CELL of CHAR] Names of the fields that should be copied from
%      the 'source' struct S (into a 'target' struct T, if provided)


if nargin<3 && ~isstruct(S),
  % T= copy_fields(S, <FLD_LIST>)
  T= copy_fields([], T, S);
  return;
end
  
if nargin<3,
  fld_list= fieldnames(S);
end
% Let us be gracious:
if ischar(fld_list),
  fld_list= {fld_list};
end

for ii= 1:length(fld_list),
  fld= fld_list{ii};
  if ~isfield(S, fld),
    warning(sprintf('Field ''%s'' not found in struct', fld));
    continue;
  end
  T.(fld)= S.(fld);
end
