function e= get_element(aa, ii)

if iscell(ii),
  S.subs= ii;
else
  S.subs= {ii};
end

if iscell(aa),
  S.type= '{}';
else
  S.type= '()';
end
e= subsref(aa, S);
