fs= 100;
freq= [9.6 10.3];
ampl= [1 1];
ampl_noise= 0.25;

T= 200;
nn= 100;
z= [[1;0]*ones(1,nn), [0;1]*ones(1,nn)];
N= size(z,2);
s= zeros(T, N);
for n= 1:N,
  ph= rand*2*pi;
  ph_jit= ph + 0.1*randn(T,1);
  zn= find(z(:,n));
  freq_jit= freq(zn) + 0.2*randn;
  ampl_jit= ampl(zn) + 0.2*randn(T,1);
  s(:,n)= ampl_jit .* sin((0:T-1)'/fs*freq_jit*2*pi + ph_jit) + ...
          ampl_noise * randn(T,1);
end

cl1= find(z(1,:));
cl2= find(z(2,:));

p= 5;
ar.y= z;
ar.x= zeros(p, N);
rc.y= z;
rc.x= zeros(p, N);
for n= 1:N,
  [a,e,k]= aryule(s(:,n), p);
  ar.x(:,n)= a(2:end)';
  rc.x(:,n)= k;
end

nc= 2;
for ic= 1:nc,
  subplot(nc, 2, ic*2-1);
  plot(ar.x(ic,cl1), ar.x(ic+1,cl1), 'r.');
  hold on;
  plot(ar.x(ic,cl2), ar.x(ic+1,cl2), 'g.');
  hold off; title('AR');
  subplot(nc, 2, ic*2);
  plot(rc.x(ic,cl1), rc.x(ic+1,cl1), 'r.');
  hold on;
  plot(rc.x(ic,cl2), rc.x(ic+1,cl2), 'g.');
  hold off; title('RC');
end

classy= 'LDA';
nTrials= [5 10];
fprintf('AR:  '); ...
doXvalidation(ar, classy, nTrials);
fprintf('RC:  '); ...
doXvalidation(rc, classy, nTrials);




return

model.classy= 'RLDA';
model.param= [0 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.8];
classy= selectModel(rc, model, [3 10]);
doXvalidation(rc, classy, nTrials);
