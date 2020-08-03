function T= copy_subfields(T, S)
%COPY_SUBFIELDS - Copy recursively (sub-)fields from one struct into another
%
%Synopsis:
%  OUT= copy_subfields(T, S)
%
%Arguments:
%  S - [STRUCT] Source, from which (sub)fields are taken
%  T - [STRUCT] Target, into which (sub)fields are pasted

% 2011-01 Benjamin Blankertz


% This code is a bit complicated, because it should also work if (sub)fields
% of T are struct arrays.

szT= size(T);
szS= size(S);

if prod(szT)==1 && prod(szS)>1,
  T= repmat(T, szS);
  szT= szS;
elseif prod(szS)==1 && prod(szT)>1,
  S= repmat(S, szT);
  szS= szT;
end
if ~isequal(szT, szS),
  error('structs have incompatible sizes');
end

fld_list= fieldnames(S);
for k= 1:prod(szT),
  for ii= 1:length(fld_list),
    fld= fld_list{ii};
    if isfield(T, fld) && ...
          (isstruct(S(k).(fld)) && isstruct(T(k).(fld))) ...
      % recursively copy subfields
      T(k).(fld)= copy_subfields(T(k).(fld), S(k).(fld));
    else
      % overwrite field in T with field of S
      T(k).(fld)= S(k).(fld);
    end
  end
end
