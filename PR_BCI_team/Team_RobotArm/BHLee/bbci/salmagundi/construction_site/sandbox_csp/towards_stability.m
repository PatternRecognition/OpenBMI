fs= 100;     %% sampling frequency
freq= [11.1 9.7 12.2 10.3];  %% frequency of signals
T= 50;      %% length of signal [samples] for each subinterval
N= 20;      %% number of subintervals per class

amp= [ 6  6  2  2;
      20  5  2  1;
       2  2  6  6;
       2  1 20  5];

col= cmap_rainbow(4);

nChans= size(amp, 1);

freq= freq + 0.01*randn(size(freq));

x= zeros(4*T, nChans);
for ii= 1:nChans,
  iv= 1:T;
  for jj= 1:2,
    amps= amp(ii,2*jj+[-1 0]);
    for kk= 1:N,
      ph= rand*2*pi;
      x(iv,ii)= amps(mod(kk,2)+1) * sin((1:T)*2*pi*freq(ii)/fs+ph)';
      iv= iv + T;
    end
  end
end

figure(1);
plotsigs(x);



idxT1= 1:T*N;
idxT2= T*N+1:2*T*N;

R1= cov(x(idxT1,:));
R2= cov(x(idxT2,:));
[W,S]= eig(R1, R1+R2);

z= x*W;
figure(2);
plotsigs(z);


xx= reshape(x, [T N*2*nChans]);
vv= var(xx);
xv= reshape(vv, [N*2 nChans]);


V1= cov(xv([1:N],:));
V2= cov(xv([1:N]+N,:));
[Wn,Sn]= eig(R1, R1+R2+V1+V2);

z= x*Wn;
figure(3);
plotsigs(z);



zv= reshape(z, [T N*2*nChans]);
zv= var(zv);
zv= reshape(zv, [N*2 nChans]);
[diag(Wn'*R1*Wn)'; diag(Wn'*R2*Wn)'; var(zv(1:N,:)); var(zv(N+1:N*2,:))]


