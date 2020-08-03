function H= axis_legend(str, hp, varargin)
% IN CONSTRUCTION

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'axis', gca, ...
                  'orientation', 'horizontal', ...
                  'dist1', 0.03, ...
                  'dist2', 0.03, ...
                  'linefrac', 0.3);

if ~strcmpi(opt.orientation, 'horizontal'),
  error('so far only orientation ''horizontal'' implemented');
end

nl= length(str);
perl= 1/nl;

d1= opt.dist1;
d2= opt.dist2;
linefrac= opt.linefrac*perl;

axes(opt.axis);
for ii= 1:nl,
  if iscell(hp),
    linestyle= hp{ii};
  else
    [dmy, linestyle]= opt_extractPlotStyles(get(hp(ii)));
    linestyle= cat(2, linestyle, {'Color', get(hp(ii), 'Color')});
  end
  H(ii).line= line(perl*(ii-1)+[d1 perl-d1-(1-d1)*linefrac], [0 0]);
  set(H(ii).line, linestyle{:});
  H(ii).text= text(perl*ii-d1-(1-d1)*linefrac+d2, 0, str{ii});
end
set(opt.axis, 'XLim',[0 1], 'box','on', 'XTick',[], 'YTick',[]);
