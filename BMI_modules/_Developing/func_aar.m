function dat=func_aar(dat, Mode, order, varargin)

if isempty(dat)
    warning('[OpenBMI] Warning! data is empty.');
end
if isempty(Mode)
    warning('[OpenBMI] Warning! Mode is empty.');
end
if isempty(order)
    warning('[OpenBMI] Order is not exist.');
    order=[10,0];
end

[T, nEvents , nChans]= size(dat.x);


tp_ar= [];
% order=5;
Dat=reshape(dat.x, [T, nEvents*nChans]);

for i= 1:nChans*nEvents,    
  aar1=aar(Dat(:,i),Mode,order);
  tp_aar(:,i)=aar1(end,:);
  end

Dat=permute(reshape(tp_aar, [order, nEvents,nChans]), [1 3 2])
Dat=reshape(Dat, [order* nChans nEvents])

dat.x= Dat;

