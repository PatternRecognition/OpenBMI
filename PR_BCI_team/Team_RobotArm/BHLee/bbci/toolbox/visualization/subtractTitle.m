function subtractTitle(VERTICAL);
%subtractTitle(<vert=0>);

if ~exist('VERTICAL', 'var'),
  VERTICAL= 0;
elseif ischar(VERTICAL),
  VERTICAL= strmatch(VERTICAL, {'horizontal','vertical'})-1;
end

if VERTICAL,
  borderSize= 0.06;
  posShrink= [1-borderSize 1 1-borderSize 1];
  posShift= [1 0 0 0];
else
  borderSize= 0.1;
  posShrink= [1 1-borderSize 1 1-borderSize];
  posShift= [0 0 0 0];
end

delete(gca);
hc= get(gcf, 'children');
for hi= 1:length(hc),
  if strcmp(get(hc(hi), 'type'), 'axes'),
    pos= get(hc(hi), 'position');
    set(hc(hi), 'position', posShift - (posShift-pos) ./ posShrink);
  end
end
