function strout= unfformat(strin)
%strout= unfformat(strin)

if iscell(strin),
  strout= apply_cellwise(strin, 'unfformat');
  return
end

iSave= find(strin=='%');
strout= strin;
for is= iSave(end:-1:1),
  strout= [strout(1:is-1) '%' strout(is:end)];
end
