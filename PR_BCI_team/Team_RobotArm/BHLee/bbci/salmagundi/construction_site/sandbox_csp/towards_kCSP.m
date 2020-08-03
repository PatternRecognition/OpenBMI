rad= [0.2 1];
T= 200;

nClusters= length(rad);
z= zeros(2, T*nClusters);
z(:,1:T)= rad(1)*randn(2,T);
ang= rand(1,T)*pi + pi/4;
z(:,T+1:end)= [cos(ang); sin(ang)].*repmat(rand(1,T)+rad(2), 2, 1);

figure(1); clf;
col= get(gca, 'colorOrder');
subplot(231); hold on;
iv= 1:T;
for k= 1:nClusters,
  plot(z(1,iv), z(2,iv), '.', 'color', col(k,:));
  iv= iv+T;
end

R1= cov(z(:,1:T)'); %R1= R1/trace(R1);
R2= cov(z(:,T+1:end)'); %R2= R2/trace(R2);
[U, D]= eig(R1+R2); 
P= diag(1./sqrt(diag(D)))*U';

S1= P*R1*P';
S2= P*R2*P';
[B,D]= eig(S1);
[B2,D2]= eig(S2); %% --> B2==B, D2==eye(2)-D;
W= B'*P;
max(diag(D), 1-diag(D))'


zz= W*z;
subplot(234); hold on;
iv= 1:T;
for k= 1:nClusters,
  plot(zz(1,iv), zz(2,iv), '.', 'color', col(k,:));
  iv= iv+T;
end

xMin= min(z(1,:));
xMax= max(z(1,:));
yMin= min(z(2,:));
yMax= max(z(2,:));
[xt,yt]= meshgrid(linspace(xMin,xMax,20), linspace(yMin,yMax,20));
test= [xt(:)'; yt(:)'];
tW= W*test;

nBins= 20;
for d= 1:2,
  subplot(2,3,1+d); hold on
  pRng= linspace(min(zz(d,:)), max(zz(d,:)), nBins);
  N= zeros(nBins, nClusters);
  iv= 1:T;
  for k= 1:nClusters,
    N(:,k)= hist(zz(d,iv), pRng)';
    iv= iv+T;
  end
  bar(pRng, N, 'stacked');
  
  subplot(2,3,4+d);
  pcolor(xt, yt, reshape(tW(d,:),size(xt)));
  shading interp;
  hold on;
  plot(z(1,:), z(2,:), 'r.');
  contour(xt, yt, reshape(tW(d,:),size(xt)), 'k');
  hold off;
end

