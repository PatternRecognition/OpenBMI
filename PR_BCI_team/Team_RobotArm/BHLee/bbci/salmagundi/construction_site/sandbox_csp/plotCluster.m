function plotCluster(z, col)
%plotCluster(z, <col>)

if nargin<2, col='b'; end
if size(z,1)<size(z,2), z=z'; end

washold= ishold;

plot(z(:,1), z(:,2), '.', 'color', col);
coz= cov(z);
%% the following line is just to make sure that principal axis are 
%% drawn properly for 'almost' spherical clusters

coz(find(abs(coz)<min(abs(diag(coz)))/1000))= 0;
[V0,D0]= eig(coz);
hold on;
plotPAxis(V0, D0, mean(z));

if ~washold, hold off; end
