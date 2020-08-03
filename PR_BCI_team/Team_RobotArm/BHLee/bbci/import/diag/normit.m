function V=normit(V)

[n,m]=size(V);

for k=1:n,
   V(:,k)=V(:,k)/norm(V(:,k),'fro');
end

