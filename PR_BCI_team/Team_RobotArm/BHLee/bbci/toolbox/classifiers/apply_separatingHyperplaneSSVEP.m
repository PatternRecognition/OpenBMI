function out= apply_separatingHyperplaneSSVEP(C, y)
%out= apply_separatingHyperplane(C, y)

Ncl=size(C,2);
Nch=size(C{1}.w,1);
ind = find(sum(abs(y),1)~=inf);
y=reshape(y,Nch,Ncl);
for i=1:Ncl
    out(i)=C{i}.w'*squeeze(y(:,i)) + C{i}.b;
end
out=-out';
%out([1 3 4 6 7 8])'