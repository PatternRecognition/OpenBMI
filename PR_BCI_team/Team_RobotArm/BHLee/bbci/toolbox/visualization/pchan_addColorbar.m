function H= pchan_addColorbar(rv, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'colormap',colormap, ...
                  'axes', 'shift', ...
                  'shiftaxes', gca, ...
                  'axsize', 0.05, ...
                  'axshift', [], ...
                  'barpos', 'auto', ...
                  'orientation', 'horiz', ...
                  'colax', 'range');

if isempty(strmatch('horiz', opt.orientation)),
  error('only orientation horizontal is implemented so far');
end

if isequal(opt.axes, 'shift'),
  if isempty(opt.axshift),
    shift= max(opt.axsize*2, 0.04);
  else
    shift= opt.axshift;
  end
  pos= get(opt.shiftaxes(1), 'Position');
  newpos= pos;
  newpos(2)= pos(2) + shift;
  newpos(4)= pos(4) - shift;
  set(opt.shiftaxes(1), 'Position', newpos);
  top= pos(2)+pos(4);
  for hi= opt.shiftaxes(2:end),
    po= get(hi, 'Position');
    newpo= po;
    newpo(2)= top - (top-po(2))*newpos(4)/pos(4);
    newpo(4)= po(4)*newpos(4)/pos(4);
    set(hi, 'Position', newpo);
  end
  cbpos= pos;
  cbpos(4)= opt.axsize;
  H.ax= axes('Position',cbpos);
elseif isnumeric(opt.axes),
  H.ax= opt.axes;
  cbpos= get(H.ax, 'Position');
end

if isstruct(rv),
  rv= rv.x(:);
end

if isnumeric(opt.colax),
  minr= opt.colax(1);
  maxr= opt.colax(2);
else
  switch(lower(opt.colax)),
   case 'range',
    minr= min(rv(:));
    maxr= max(rv(:));
   case 'sym',
    maxr= max(abs(rv(:)));
    minr= -maxr;
   case '0tomax',
    minr= 0;
    maxr= max(abs(rv(:)));
   otherwise,
    error('invalid choice for colax');
  end
end

if minr==0,
  ht1= text(0, 0.6, sprintf('0  '));
else
  ht1= text(0, 0.6, sprintf('%.3f  ', minr));
end
ht2= text(1, 0.6, sprintf('  %.3f [r^2]', maxr));
set(ht2, 'HorizontalAli','right');

if isequal(opt.barpos, 'auto'),
  ext1= get(ht1, 'Extent');
  ext2= get(ht2, 'Extent');
  xx= [ext1(1)+ext1(3) ext2(1)];
else
  xx= opt.barpos;
end
impos= cbpos;
impos([1 3])= [impos(1)+impos(3)*xx(1) impos(3)*(xx(2)-xx(1))];
H.axim= axes('Position',impos);

image_local_cmap(1:size(opt.colormap,1), opt.colormap);
axis_redrawFrame;

set(H.ax, 'Visible','off');
