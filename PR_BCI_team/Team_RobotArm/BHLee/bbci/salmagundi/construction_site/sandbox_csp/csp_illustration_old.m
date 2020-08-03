fig_dir= 'augcog03/';

pos= [-0.5 -0.25; 0.75 1]';
%pos= zeros(2,2); 
siz= [1 .2; .3 .8]';
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


R1= cov(z(:,1:T)');     %R1= R1/trace(R1);
R2= cov(z(:,T+1:end)'); %R2= R2/trace(R2);
[U, L]= eig(R1+R2);
P= diag(1./sqrt(diag(L)))*U';

S1= P*R1*P';
[B,D]= eig(S1);
S2= P*R2*P';
[B2,D2]= eig(S2); 
%% --> abs(B2)==abs(B), 
%%     and if R1, R2 have normalized traces: D2==eye(2)-D;
W= B'*P;
%if D(1,1)<D(2,2), W= flipud(W); end


clf;
col= [0.9 0 0; 0 0.7 0];
%set(gca, 'colorOrder',col);
ha(1)= subplot(131); hold on;
iv= 1:T;
for k= 1:nClusters,
  plotCluster(z(:,iv), col(k,:));
  iv= iv+T;
end
plotPAxis(U, L, [0 0], 'b');
hold off;
axis square; 
axisequalwidth;

zp= P*z;
ha(2)= subplot(132); hold on;
iv= 1:T;
for k= 1:nClusters,
  plotCluster(zp(:,iv), col(k,:));
  iv= iv+T;
end
hold off;
axis square;
axisequalwidth;

zw= W*z;
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

saveFigure([fig_dir 'csp_demo'], [16 5]*1.2);
