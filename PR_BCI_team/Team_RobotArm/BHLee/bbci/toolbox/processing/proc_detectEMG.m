function dat= proc_detectEMG(dat, nSeg)
%dat= proc_detectEMG(dat, <nSeg=2>)

if nargin<2, nSeg= 2; end

[T, nChans, nMotos]= size(dat.x);
yMa= movingAverageCausal(dat.x(:,:), 2);
yMa= yMa(2:end,:);

y1= max(abs(diff(yMa)));
xo= reshape(y1, [nChans nMotos]);

y1= std(dat.x(:,:));
xo= [xo; reshape(y1, [nChans nMotos])];

if nSeg>0,
  inter= round(linspace(1, T, nSeg+1));
  for si= 1:nSeg,
    iv= inter(si):inter(si+1);
    yi= std(dat.x(iv,:));
    xo= [xo; reshape(yi, [nChans nMotos])];
  end
end

dat.x= xo;
