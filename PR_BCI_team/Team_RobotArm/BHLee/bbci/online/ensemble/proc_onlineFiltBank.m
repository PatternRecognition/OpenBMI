function DAT=proc_onlineFiltBank(dat,b_array,a_array)
%  dat.x needs dimentions: time x channel

%data=dat.x;
%size(data);
DAT.x=[];
%size(a_array);
for i=1:size(a_array,1)
  a=a_array(i,:);
  b=b_array(i,:);
  x(:,:)= filter(b,a,dat.x(:,:));
  DAT.x=cat(2,DAT.x,x);  
end
DAT.clab=dat.clab;