function clab= getClabFromJoinlist(join)

clab= {};
nJoins= length(join);
for ij= 1:nJoins,
  joStr= join{ij};
  is= find(joStr=='+' | joStr=='-');
  if isempty(is),
    clab= cat(2, clab, {joStr});
  else
    ch1= joStr(1:is-1);
    ch2= joStr(is+1:end);
    clab= cat(2, clab, {ch1}, {ch2});
  end
end
clab= unique(clab);
