function strout= untex(strin)
%strout= untex(strin)

if iscell(strin),
  strout= apply_cellwise(strin, 'untex');
  return
end

iSave= find(ismember(strin, '_^\%&#'));
strout= strin;
for is= iSave(end:-1:1),
  strout= [strout(1:is-1) '\' strout(is:end)];
end
