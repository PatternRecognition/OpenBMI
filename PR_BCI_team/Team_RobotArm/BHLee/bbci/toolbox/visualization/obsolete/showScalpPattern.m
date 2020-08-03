function [h_ax, hc, h]= showScalpPattern(mnt, w, SHOW_LABELS, ...
                                         SCALE_POS, COL_AX, CONTOUR, OPT)
%h_ax= showScalpPattern(mnt, w, <showLabels=1, scalePos, COL_AX, CONTOUR=0>)
%
% scalePos: 'horiz', 'vert', 'none'

% bb, GMD.FIRST.IDA, 08/00

bbci_obsolete(mfilename, 'scalpPlot');

if ~exist('SHOW_LABELS','var') | isempty(SHOW_LABELS), SHOW_LABELS=1; end
if ~exist('SCALE_POS','var') | isempty(SCALE_POS), SCALE_POS='vert'; end
if ~exist('COL_AX','var'), COL_AX=[]; end
if ~exist('CONTOUR','var') | isempty(CONTOUR), CONTOUR=0; end
if ~exist('OPT','var'), OPT=[]; end
 if ~isfield(OPT, 'markerSize'), OPT.markerSize= 20; end
 if ~isfield(OPT, 'fontSize'), OPT.fontSize= 8; end
 if ~isfield(OPT, 'minorFontSize'), OPT.minorFontSize= 6; end
 if ~isfield(OPT, 'crossSize'), OPT.crossSize= 2; end


w= w(:);
if length(w)==length(mnt.clab),
  dispChans= find(~isnan(mnt.x) & ~isnan(w));
  w= w(dispChans);
else
  dispChans= find(~isnan(mnt.x));
  if length(w)~=length(dispChans),
    error(['length of w must match # of displayable channels, ' ...
           'i.e. ~isnan(mnt.x), in mnt']);
  end
end
xe= mnt.x(dispChans);
ye= mnt.y(dispChans);

oldUnits= get(gca, 'units');
set(gca, 'units', 'normalized');
pos= get(gca, 'position');
set(gca, 'units', oldUnits);
resolution= max(20, 60*pos(3));
xx= linspace(min(xe), max(xe), resolution);
yy= linspace(min(ye), max(ye), resolution)';
[xg,yg,zg]= griddata(xe, ye, w, xx, yy, 'linear');

pcolor(xg, yg, zg);
h_ax= gca;
if isempty(COL_AX),
  zgMax= max(abs(caxis));
  COL_AX= [-zgMax zgMax];
elseif isequal(COL_AX, 'range'),
  COL_AX= [min(caxis) max(caxis)];
end
%if COL_AX,
  caxis(COL_AX);
%end
shading interp;

hold on;
if CONTOUR,
  if CONTOUR>0,
    v= floor(min(COL_AX)):CONTOUR:ceil(max(COL_AX));
  else
    v= linspace(min(COL_AX), max(COL_AX), -CONTOUR+2);
  end
  v([1 end])= []
  if length(v)>1,  %% ~isempty(v),
    [c,h]= contour(xg, yg, zg, v, 'k-');
    set(h, 'linewidth',1);
%    if length(h)>1,
%      clabel(c,h);
%    end
  end
end

T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
plot(xx, yy, 'k');
nose= [1 1.1 1];
nosi= [86 90 94]+1;
plot(nose.*xx(nosi), nose.*yy(nosi), 'k');

if SHOW_LABELS,
  labs= {mnt.clab{dispChans}};
  plot(xe, ye,'ko', 'markerSize',OPT.markerSize);
  h= text(xe, ye, labs);
  set(h, 'fontSize',OPT.fontSize, 'horizontalAlignment','center');
  for il= 1:length(labs),
    strLen(il)= length(labs{il});
  end
  iLong= find(strLen>3);
  set(h(iLong), 'fontSize',OPT.minorFontSize);
else
  h= plot(xe, ye, 'k+');
  set(h, 'markerSize',OPT.crossSize);
end

%box off;
hold off;
set(gca, 'xTick',[], 'yTick',[]); %, 'xColor','w', 'yColor','w');
axis('xy', 'tight', 'equal', 'tight');
if ~strcmp(SCALE_POS, 'none'),
  hc= colorbar(SCALE_POS);
end
axis('off');
