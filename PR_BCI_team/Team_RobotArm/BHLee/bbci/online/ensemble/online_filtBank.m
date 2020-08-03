function DAT=online_filtBank(dat,b_array,a_array)
%  dat.x needs dimentions: time x channel

%data=dat.x;
%size(data);
DAT.x=zeros(size(a_array,1),size(dat.x,1),size(dat.x,2));
%size(a_array);
for i=1:size(a_array,1)
  a=a_array(i,:);
  b=b_array(i,:);
  x(:,:)= filter(b,a,dat.x(:,:));
  DAT.x(i,:,:)=x;  
end
DAT.clab=dat.clab;