function [H, v]= plotScalpPattern(mnt, w, varargin);
%H= plotScalpPattern(mnt, w, <opt>)
%
% possible fields of 'opt' are
%     .showLabels:  0 or 1
%     .scalePos:    'horiz', 'vert', 'none'
%     .colAx:       'range', 'sym', or [minVal maxVal]
%      markerSize, fontSize, minorFontSize, crossSize,
%      contour, resolution, shading
%     .colormap

% bb, GMD.FIRST.IDA, 08/00

bbci_obsolete(mfilename, 'scalpPlot');

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'markerSize', 20, ...
                  'fontSize', 8, ...
                  'minorFontSize', 6, ...
                  'crossSize', 2, ...
                  'contour', 0, ...
                  'contour_policy', 'choose', ...
                  'contour_lineprop', {}, ...
                  'shading', 'interp', ...
                  'resolution', [], ...
                  'colAx', 'range', ...
                  'scalePos', 'vert', ...
                  'showLabels', 0, ...
                  'linespec', {'k'}, ...
                  'mark_channels', [], ...
                  'mark_properties', {'lineWidth',3, 'markerSize',10});

if isfield(opt, 'colormap'),
  colormap(opt.colormap);
end

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

H.ax= gca;
oldUnits= get(H.ax, 'units');
set(H.ax, 'units', 'normalized');
pos= get(H.ax, 'position');
set(H.ax, 'units', oldUnits);
if isempty(opt.resolution), opt.resolution= max(20, 60*pos(3)); end
st= (max(xe)-min(xe))/opt.resolution;
xx= min(xe)-st/2:st:max(xe)+st/2;
st= (max(ye)-min(ye))/opt.resolution;
yy= (min(ye)-st/2:st:max(ye)+st/2)';
[xg,yg,zg]= griddata(xe, ye, w, xx, yy, 'linear');

H.patch= pcolor(xg, yg, zg);
tight_caxis= caxis;
if isequal(opt.colAx, 'sym'),
  zgMax= max(abs(tight_caxis));
  opt.colAx= [-zgMax zgMax];
elseif isequal(opt.colAx, 'range'),
  opt.colAx= [min(tight_caxis) max(tight_caxis)];
end
%if opt.colAx,
  caxis(opt.colAx);
%end
shading(opt.shading);

hold on;
if ~isequal(opt.contour,0),
  if length(opt.contour)>1,
    v= opt.contour;
%    v= v(min(find(v>min(opt.colAx))):max(find(v<max(opt.colAx))));
  else
    mi= min(tight_caxis);
    ma= max(tight_caxis);
    switch(opt.contour_policy),
     case 'strict',
      if opt.contour>0,
        v= floor(mi):opt.contour:ceil(ma);
      else
        v= linspace(mi, ma, -opt.contour+2);
      end
      v([1 end])= [];
     case 'choose',
      v= goodContourValues(mi, ma, opt.contour);
     otherwise
      error('contour_policy not known');
    end
  end
  if length(v)>1,  %% ~isempty(v),
    [c,H.contour]= contour(xg, yg, zg, v, 'k-');
    set(H.contour, 'linewidth',1, opt.contour_lineprop{:});
%    if length(h)>1,
%      clabel(c,h);
%    end
  end
end

T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
plot(xx, yy, opt.linespec{:});
nose= [1 1.1 1];
nosi= [86 90 94]+1;
H.nose= plot(nose.*xx(nosi), nose.*yy(nosi), opt.linespec{:});

opt.mark_channels= chanind(mnt.clab(dispChans), opt.mark_channels);
if opt.showLabels,
  labs= {mnt.clab{dispChans}};
  H.label_markers= plot(xe, ye,'ko', 'markerSize',opt.markerSize);
  H.label_text= text(xe, ye, labs);
  set(H.label_text, 'fontSize',opt.fontSize, 'horizontalAlignment','center');
  strLen= apply_cellwise(labs, 'length');
  strLen = [strLen{:}];
  iLong= find(strLen>3);
  set(H.label_text(iLong), 'fontSize',opt.minorFontSize);
  if ~isempty(opt.mark_channels),
    set(H.label_text(opt.mark_channels), 'fontWeight','bold');
  end
else
  if opt.crossSize>0,
    H.cross= plot(xe, ye, 'k+', 'markerSize',opt.crossSize);
  end
  if ~isempty(opt.mark_channels),
    H.cross_marked= plot(xe(opt.mark_channels), ye(opt.mark_channels), ...
                         'k+', opt.mark_properties{:});
  end
end

%box off;
hold off;
set(H.ax, 'xTick',[], 'yTick',[]); %, 'xColor','w', 'yColor','w');
axis('xy', 'tight', 'equal', 'tight');
if strcmp(opt.scalePos, 'none'),
  H.cb= [];
else,
  H.cb= colorbar(opt.scalePos);
  if length(opt.contour)>1,
    set(H.cb, 'yTick',opt.contour);
  end
end
axis('off');

if nargout==0,
  clear H;
end
