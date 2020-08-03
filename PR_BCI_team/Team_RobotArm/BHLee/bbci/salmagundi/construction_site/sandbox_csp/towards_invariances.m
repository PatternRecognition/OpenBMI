fs= 100;     %% sampling frequency
freq= [11.1 9.7 12.2 10.3];  %% frequency of signals
T= 400;      %% length of signal [samples] for each subcondition

amp= [10 10  2  2;
      20  5  3  1;
       2  2 10 10;
       3  1 20  5];

col= cmap_rainbow(4);

nChans= size(amp, 1);

freq= freq + 0.01*randn(size(freq));

x= zeros(4*T, nChans);
for ii= 1:nChans,
  iv= 1:T;
  for jj= 1:4,
    ph= rand*2*pi;
    x(iv,ii)= amp(ii,jj) * sin((1:T)*2*pi*freq(ii)/fs+ph)';
    iv= iv + T;
  end
end

figure(1);
plotsigs(x);

xx= reshape(x, [T 4 nChans]);
xx= permute(xx, [1 3 2]);
y_target= [1 1 0 0];
y_distrub= [1 0 1 0];

fv1= struct('x',xx, 'y',[y_target; 1-y_target]);
fv2= struct('x',xx, 'y',[y_distrub; 1-y_distrub]);

[fc1,W1,l1]= proc_csp3(fv1);
[fc2,W2,l2]= proc_csp3(fv2);



z= reshape(ipermute(fc1.x, [1 3 2]), [T*4 nChans]);
figure(2);
plotsigs(z);

z= reshape(ipermute(fc2.x, [1 3 2]), [T*4 nChans]);
figure(3);
plotsigs(z);



idxT1= 1:2*T;
idxT2= 2*T+1:4*T;
idxD1= [1:T, 2*T+1:3*T];
idxD2= [T+1:2*T, 3*T+1:4*T];

R1= cov(x(idxT1,:));
R2= cov(x(idxT2,:));
[W,S]= eig(R1, R1+R2);

z= x*W;
figure(2);
plotsigs(z);


N1= cov(x(idxD1,:));
N2= cov(x(idxD2,:));
[Wn,Sn]= eig(R1, R1+R2+N1);

z= x*Wn;
figure(3);
plotsigs(z);


N= cov(x([idx1 idxD2],:));
[Wn,Sn]= eig(R1, R1+R2+N);

z= x*Wn;
figure(3);
plotsigs(z);
