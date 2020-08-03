function [out]= online_linearDerivation(dat, A, Nfilter)
%out= online_linearDerivation(dat, A, <clab>)
%loop over the repmat:
size(dat.x);
out=dat;
DAT=zeros(size(dat.x,2),size(dat.x,3),sum(Nfilter));

A=permute(A,[2 3 1]);
dat.x=permute(dat.x,[2 3 1]);
%size(DAT)
index=cumsum(Nfilter);
ii=1;jj=1;
while jj <= index(end)
  if jj>index(ii)
    ii=ii+1;
  end
  DAT(:,:,jj)=dat.x(:,:,ii);
  %DAT=cat(3,DAT,dum);
  %size(DAT)
  jj=jj+1;
end
%A=A(:,1:37,:);
i=1;
%size(dat.x(:,:,i));
%size(A);
out.x=zeros(size(A,3),size(A,1),size(dat.x,1));
for i=1:size(A,3)
 out.x(i,:,:)=DAT(:,:,i)*A(:,:,i)';
  end
out.x=permute(out.x,[2 3 1]);
out.x=reshape(out.x,size(out.x,1),size(out.x,2)*size(out.x,3));