addpath([BCI_DIR 'diag']);

pos= [-0.5 -0.25; 0.75 1; -0.75 1.1]';
%pos= zeros(2,2); 
siz= [1 .2; .3 .8; 1.1 0.3]';
T= 200;

nClusters= size(pos, 2);
z= zeros(2, T*nClusters);
iv= 1:T;
for k= 1:nClusters,
  psi= pi*rand;
  R= [cos(psi) -sin(psi); sin(psi) cos(psi)];
  z(:,iv)= repmat(pos(:,k), 1, T) + R*(randn(2, T).*repmat(siz(:,k),1,T));
  iv= iv+T;
end

R= zeros(2, 2, nClusters);
D= zeros(2, 2, nClusters);

iv= 1:T;
for k= 1:nClusters,
  R(:,:,k)= cov(z(:,iv)');
  iv= iv+T;
end

[P,dmy]= simDiag(R, 'fastdiag');
for k= 1:nClusters,
  D(:,:,k)= P'*R(:,:,k)*P;
end
W= P*diag(1./sqrt(diag(sum(D,3))));

clf;
col= [0.9 0 0; 0 0.7 0; 0 0 1];
%set(gca, 'colorOrder',col);
ha(1)= subplot(131); hold on;
iv= 1:T;
for k= 1:nClusters,
  plotCluster(z(:,iv), col(k,:));
  iv= iv+T;
end
hold off;
axis square; 
axisequalwidth;

zp= P'*z;
ha(2)= subplot(132); hold on;
iv= 1:T;
for k= 1:nClusters,
  plotCluster(zp(:,iv), col(k,:));
  iv= iv+T;
end
hold off;
axis square;
axisequalwidth;

zw= W'*z;
ha(3)= subplot(133); hold on;
iv= 1:T;
for k= 1:nClusters,
  plotCluster(zw(:,iv), col(k,:));
  iv= iv+T;
end
hold off;
axis square; 
axisequalwidth;

set(ha, 'xTick',[], 'yTick',[], 'box','on');
