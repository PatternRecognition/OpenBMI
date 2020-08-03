function H= showScalpSeries(epo, mnt, inter, opt)
%showScalpSeries(epo, mnt, inter, <opt>)
%
% the opt argument is passed to showScalpPattern

bbci_obsolete(mfilename, 'scalpEvolution');

if ~exist('opt','var'), opt.showLabels=0; end
%opt.colAx= 'range';
opt.scalePos= 'none';

nSteps= length(inter)-1;
nClasses= size(epo.y, 1);

clf;
clInd= find(epo.y(1,:));
h= zeros(nSteps,1);
for is= 1:nSteps,
  h(is)= suplot(nSteps+1, is);
  iv= getIvalIndices(inter(is:is+1), epo);
  w= mean(mean(epo.x(iv, :, clInd)), 3);
  plotScalpPattern(mnt, w, opt);
  axis off
  title(sprintf('[%g %g] ms', trunc(inter(is:is+1))));
end
unifyCLim(h);

if isfield(epo, 'yUnit'),
  ylab= ['[' epo.yUnit ']'];
else
  ylab= '[\muV]';
end

hc= suplot(nSteps+1, nSteps+1);
colPos= get(hc, 'position');
delete(hc);
hp= gca;
oldPos= get(hp, 'position');
hc= colorbar('vert');
axes(hc);
ylabel(ylab);
set(hc, 'position', colPos);
set(hp, 'position', oldPos);

if isfield(epo, 'title'),
  addTitle(untex(epo.title), 1, 0);
end
