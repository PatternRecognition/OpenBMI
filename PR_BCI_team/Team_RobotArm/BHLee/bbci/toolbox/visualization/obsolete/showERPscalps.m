function H= showERPscalps(epo, mnt, inter, unifiedCLim, opt)
%showERPscalps(epo, mnt, inter, <unifiedCLim=1, opt>)
%
% the opt argument is passed to plotScalpPattern

bbci_obsolete(mfilename, 'scalpEvolution');

if ~exist('unifiedCLim','var') | isempty(unifiedCLim), unifiedCLim=1; end
if ~exist('opt','var'), opt.showLabels=0; end
%opt.colAx= 'range';
opt.scalePos= 'none';
if ~isfield(opt, 'xUnit'), 
  if isfield(epo, 'xUnit'),
    opt.xUnit= epo.xUnit;
  else
    opt.xUnit= 'ms'; 
  end
end

nSteps= length(inter)-1;
nClasses= size(epo.y, 1);
epo= proc_selectChannels(epo, mnt.clab(find(~isnan(mnt.x))));
mnt= mnt_restrictMontage(mnt, epo.clab);

clf;
H= [];
for ic= 1:nClasses,
  clInd= find(epo.y(ic,:));
  h= [];
  for is= 1:nSteps,
    h= [h subplot(nClasses, nSteps, is+(ic-1)*nSteps)];
    iv= getIvalIndices(inter(is:is+1), epo);
    w= mean(mean(epo.x(iv, :, clInd), 1), 3);
    plotScalpPattern(mnt, w, opt);
    axis off
%    axis on
%    if is==1 & isfield(epo, 'className'),
%      ylabel(epo.className{ic}, 'color','k');
%    end
    if ic==1,
      title(sprintf('[%g %g] %s', trunc(inter(is:is+1)), opt.xUnit));
    end
  end
  if length(unifiedCLim)==2,
    set(h, 'cLim',unifiedCLim);
  else
    unifyCLim(h);
  end
  H= [H h];
end

if isequal(unifiedCLim,1), unifyCLim; end

if isfield(epo, 'yUnit'),
  ylab= ['[' epo.yUnit ']'];
else
  ylab= '[\muV]';
end

for ic= 1:nClasses,
  hp= subplot(nClasses, nSteps, (ic-1)*nSteps+1);
  pos= get(gca, 'pos');
  xp(ic)= 0.5*pos(1);
  yp(ic)= pos(2)+0.5*pos(4);

  hp= subplot(nClasses, nSteps, ic*nSteps);
  oldPos= get(hp, 'position');
  hc= colorbar('vert');
  axes(hc);
  ylabel(ylab);
  colPos= get(hc, 'position');
  colPos(1)= 0.92;
  set(hc, 'position', colPos);
  set(hp, 'position', oldPos);
end
ax= axes('position', [0 0 1 1]);
ht= text(xp, yp, epo.className);
set(ht, 'horizontalAli','center', 'verticalAli','middle', ...
        'rotation',90, 'fontSize',14);
set(ax, 'visible','off');

if isfield(epo, 'title'),
  addTitle(untex(epo.title), 1, 0);
end
