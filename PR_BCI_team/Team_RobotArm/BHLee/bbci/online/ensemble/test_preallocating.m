iter=200;
A=randn(300,20);
B=[];
C=zeros(iter,size(A,1),size(A,2));
tic
for i=1:iter
  B=cat(1,A,B);
%  size(B)
end
toc
clear B
tic
for i=1:iter
  C(i,:,:)=A;
 % size(C)
end
toc