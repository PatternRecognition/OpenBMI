function H= grid_addScale(mnt, varargin)
% experimental, called by grid_plot

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'xUnit', 'ms', ...
                  'yUnit', '\muV', ...
                  'scaleHPos', 'zeroleft', ...
                  'scaleVPos', 'zeromiddle', ...
                  'scaleFontSize', get(gca, 'fontSize'), ...
                  'scaleShowOrientation', 1);

xLim= get(gca, 'xLim');
yLim= get(gca, 'yLim');
xt= get(gca, 'xTick');
xx= median(diff(xt));
yt= get(gca, 'yTick');
yy= median(diff(yt));

x= [mnt.box(1,:) mnt.scale_box(1)];
y= [mnt.box(2,:) mnt.scale_box(2)];
w= [mnt.box_sz(1,:) mnt.scale_box_sz(1)];
h= [mnt.box_sz(2,:) mnt.scale_box_sz(2)];
bs= 0.005;
siz= (1+2*bs)*[max(x+w) - min(x) max(y+h) - min(y)];
pos= [mnt.scale_box(1)-min(x) mnt.scale_box(2)-min(y) ...
      mnt.scale_box_sz(1) mnt.scale_box_sz(2)]./[siz siz];
pos= pos + [bs bs 0 0];
H.ax= axes('position', pos);
ud= struct('type','ERP plus', 'chan','scale');
set(H.ax, 'xLim',xLim, 'yLim',yLim, 'userData',ud);

if strncmp('zero', opt.scaleHPos, 4),
  if xLim(1)<0-0.1*diff(xLim) & xLim(2)>xx,
    opt.scaleHPos= 'Zero';
  else
    opt.scaleHPos= opt.scaleHPos(5:end);
    if isempty(opt.scaleHPos), opt.scaleHPos= 'middle'; end
  end
end
if strncmpi('zero', opt.scaleVPos, 4),
  if yLim(1)<0-0.1*diff(yLim) & yLim(2)>yy+0.1*diff(yLim),
    opt.scaleVPos= 'Zero';
  else
    opt.scaleVPos= opt.scaleVPos(5:end);
    if isempty(opt.scaleVPos), opt.scaleVPos= 'middle'; end
  end
end

if isfield(opt, 'cLim')
   set(H.ax, 'cLim', opt.cLim')
   if isfield(opt, 'colormap')
      colormap(opt.colormap); 
   end
   cb = colorbar;
   opt.scaleHPos = 'left';
   ap = get(H.ax, 'position');
   set(cb, 'position', [ap(1)+0.8*ap(3) ap(2)+0.1*ap(4) 0.1*ap(3) 0.8*ap(4)])
  set(get(cb,'yLabel'),'String', opt.zUnit)
end

switch(opt.scaleHPos),
 case 'Zero',
  x0= 0;
 case 'left',
  x0= xLim(1) + 0.05*diff(xLim);
 case 'middle',
  x0= mean(xLim) - xx/2;
 case 'right',
  x0= xLim(2) - xx - 0.05*diff(xLim);
 otherwise,
  error('unimplemented choice for opt.scaleHPos');
end
switch(opt.scaleVPos),
 case 'Zero',
  y0= 0;
% case 'top',
%  y0= yLim(2),
 case 'middle',
  y0= mean(yLim) - yy/2;
% case 'bottom',
%  y0= yLim(1);
 otherwise,
  error('unimplemented choice for opt.scaleVPos');
end

H.vline= line([x0 x0], [y0 y0+yy]);
set(H.vline, 'lineWidth',2, 'color','k');
H.text_yUnit= text(x0, y0+yy/2, sprintf(' %g %s', yy, opt.yUnit));
set(H.text_yUnit, 'verticalAli','middle', 'fontSize',opt.scaleFontSize);
if opt.scaleShowOrientation,
  if strcmpi(opt.yDir, 'reverse'),
    sgn= '-';
  else
    sgn= '+';
  end
  H.text_sgn= text(x0, y0+yy, [' ' sgn]);
  set(H.text_sgn, 'verticalAli','bottom', 'fontSize',opt.scaleFontSize);
end

H.hline= line([x0 x0+xx], [y0 y0]);
set(H.hline, 'lineWidth',2, 'color','k');
yt= y0-0.03*diff(yLim);
H.text_xUnit= text(x0+xx/2, yt, ...
                   sprintf('%g %s', xx, opt.xUnit));
set(H.text_xUnit, 'horizontalAli','center', 'verticalAli','top', ...
                  'fontSize',opt.scaleFontSize);
if x0==xLim(1), %% leftmost position
  set(H.text_xUnit, 'position',[x0 yt 0], 'horizontalAli','left');
end

axis off;

if nargout==0,
  clear H;
end
