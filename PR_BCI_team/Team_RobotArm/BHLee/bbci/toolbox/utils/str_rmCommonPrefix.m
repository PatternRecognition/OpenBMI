function Cout= str_rmCommonPrefix(C)
%C= str_rmCommonPrefix(C)

Clen= length(C);
if Clen>1,
  cc= char(C);
  ii= min(find(~all(cc==repmat(cc(1,:),[size(cc,1) 1]),1)));
  Cout= cell(1, Clen);
  for mm= 1:Clen,
    Cout{mm}= C{mm}(ii:end);
  end
end
