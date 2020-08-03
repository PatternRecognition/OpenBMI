function [N, hp, hl, hleg]= showERP(epo, mnt, Chan, varargin)
%[N, hp, hl, hleg]= showERP(epo, mnt, chan, <FLAGS>)
%
% IN    epo     - struct of epoched signals, see makeSegments
%       mnt     - struct for electrode montage, see setElectrodeMontage
%       chan    - channel label or index
%       FLAGS   -
%         'diff'      plot differences between class averages
%         'var'       plot st.derivation
%         'legend'    show class legend
%         'small'     setup for small axes
%
% OUT  N       - number of averaged events
%
% SEE  makeSegments, setElectrodeMontage

% bb, GMD.FIRST.IDA 09/00


DIFF= ~isempty(strmatch('diff', {varargin{:}}));
VAR= ~isempty(strmatch('var', {varargin{:}}));
LEGEND= ~isempty(strmatch('legend', {varargin{:}}));
SMALL= ~isempty(strmatch('small', {varargin{:}}));
GRID= ~isempty(strmatch('grid', {varargin{:}}));
if GRID, SMALL=1; end
hl= [];

if ~exist('Chan','var'), Chan=1:length(epo.clab); end
if isfield(epo, 'y'),
  nClasses= size(epo.y, 1);
else
  nClasses= 1;
end
chan= chanind(epo, Chan);
nChans= length(chan);
if nChans==0,
  error('channel not found'); 
elseif nChans>1,
  col= hsv2rgb([(1:nChans)'/nChans ones(nChans,1) 0.85*ones(nChans,1)]);
  hp= [];
  for ic= 1:nChans,
    [N, hpp, hll]= showERP(epo, mnt, chan(ic));
    delete(hll);
    hp= [hp; hpp(1)];
    set(hpp, 'color',col(ic,:));
    set(hpp(2:end), 'lineStyle','--');
    hold on;
  end
  hold off;
  axis tight;
  set(gca, 'yLimMode', 'manual');
  hl= line([0 0], [-1e10 1e10], 'color','k');
  set(hl, 'handleVisibility','off');
  hleg= legend(hp, epo.clab{chan}, 0);
  if isfield(epo, 'title'),
    title(untex(epo.title));
  end
  ud= struct('type','ERP', 'chan',{Chan}, 'hleg',hleg);
  set(gca, 'userData', ud);
  return
end

T= length(epo.t);
N= zeros(nClasses, 1);
xm= zeros(T, nClasses);
for cc= 1:nClasses,
  if isfield(epo, 'y'),
    ei= find(epo.y(cc,:));
  else
    ei= 1;
  end
  N(cc)= length(ei);
  x= squeeze(epo.x(:, chan, ei));
  xm(:,cc)= mean(x, 2);
  if VAR,
    if cc==1, xs= zeros(size(xm)); end
    xs(:,cc)= std(x, 0, 2);
  end
end
if DIFF & nClasses>1,
  hp= plot(epo.t, xm(:,1)-xm(:,2));
else
  hp= plot(epo.t, xm);
end
if VAR,
  hold on;
  plot(epo.t, [xm-xs xm+xs], 'm');
  hold off;
end

set(gca, 'xLim', [epo.t(1) epo.t(end)]);
if LEGEND & ~DIFF & ~any(isnan(getAxisGridPos(mnt, 0))),
  if isfield(epo, 'className'),
    hleg= legend(hp, epo.className, 0);
  else
    hleg= legend(hp, [repmat('class ',nClasses,1) num2str((1:nClasses)')], 0);
  end
else
  hleg= NaN;
end

if SMALL & ~GRID,
  set(gca, 'xTick', 0, 'xTickLabel', [], 'xGrid', 'on', ...
           'yTickLabel', [], 'yGrid', 'on');
elseif ~GRID,
  title(epo.clab(chan));
  set(gca, 'YLimMode', 'manual');
  hl= line([0 0], [-1e10 1e10], 'color','k');
  moveObjectBack(hl);
  set(hl, 'HandleVisibility','off');
%  ym= 10*max(abs(yl));
%  line([0 0], [-ym ym], 'color','k');
%  set(gca, 'yLim',yl);
end
if isfield(epo, 'N'),
  N= epo.N;
end

ud= struct('type','ERP', 'chan',epo.clab{chan}, 'hleg',hleg);
set(gca, 'userData', ud);
