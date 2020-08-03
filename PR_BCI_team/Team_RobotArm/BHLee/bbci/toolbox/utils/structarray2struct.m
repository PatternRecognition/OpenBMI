function T= structarray2struct(S)

T= struct;
for F= fieldnames(S)',
  fld= F{1};
  for ii= 1:length(S),
    val= getfield(S, {ii}, fld);
    if isempty(val),
      vec(ii)= NaN;
    else
      vec(ii)= getfield(S, {ii}, fld);
    end
  end
  T= setfield(T, fld, vec);
end
