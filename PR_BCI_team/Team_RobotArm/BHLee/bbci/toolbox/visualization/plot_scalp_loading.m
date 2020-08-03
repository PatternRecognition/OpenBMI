function H= plot_scalp_loading(mnt, w, varargin)
%plot_scalp_loading(mnt, w, <opt>);

% bb, ida.first.fhg.de 08/2000

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'show_labels', 1, ...
                  'scale_pos', 'vert', ...
                  'fontSize', 8, ...
                  'minorFontSize', 6, ...
                  'textColor', 'k', ...
                  'colAx', 'sym', ...
                  'show_nose', 1, ...
                  'lineWidth', 2, ...
                  'lineColor', 'k', ...
                  'radius', 0.074);

tight_caxis= [min(w) max(w)];
if isequal(opt.colAx, 'sym'),
  zgMax= max(abs(tight_caxis));
  opt.colAx= [-zgMax zgMax];
elseif isequal(opt.colAx, 'range'),
  opt.colAx= [min(tight_caxis) max(tight_caxis)];
elseif isequal(opt.colAx, '0tomax'),
  opt.colAx= [0.0001 max(tight_caxis)];
end
caxis(opt.colAx);

w= w(:);
if length(w)==length(mnt.clab),
  dispChans= find(~isnan(mnt.x(:)) & ~isnan(w));
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

%% Head
H.ax= gca;
T= linspace(0, 2*pi, 360);
xx= cos(T);
yy= sin(T);
H.head= plot(xx, yy, 'k');
hold on;

%% Electrodes
T= linspace(0, 2*pi, 18);
disc_x= opt.radius*cos(T); 
disc_y= opt.radius*sin(T);

for ic= 1:length(dispChans),
  patch(xe(ic)+disc_x, ye(ic)+disc_y, w(ic));
  h= line(xe(ic)+disc_x, ye(ic)+disc_y);
  set(h, 'color',opt.lineColor, 'lineWidth',opt.lineWidth);
end
caxis(opt.colAx);


%% Nose
if opt.show_nose,
  nose= [1 1.1 1];
  nosi= [86 90 94]+1;
  H.nose= plot(nose.*xx(nosi), nose.*yy(nosi), 'k');
end

%% Labels
if opt.show_labels,
  labs= {mnt.clab{dispChans}};
  H.label_text= text(xe, ye, labs);
  set(H.label_text, 'horizontalAlignment','center', ...
         'fontSize',opt.fontSize, 'color',opt.textColor);
  strLen= apply_cellwise(labs, 'length');
  iLong= find([strLen{:}]>3);
  set(H.label_text(iLong), 'fontSize',opt.minorFontSize);
end

hold off;
set(H.ax, 'xTick', [], 'yTick', []);
axis('xy', 'tight', 'equal', 'tight', 'off');

if nargout==0,
  clear H;
end
