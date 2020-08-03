function [h, ax]= addTitle(tit, VERTICAL, shift);
%[h, ax]= addTitle(tit, <vert=0, shift=1>);

if ~exist('shift', 'var'), shift=1; end
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

if shift,
  hc= get(gcf, 'children');
  for hi= 1:length(hc),
    if strcmp(get(hc(hi), 'type'), 'axes'),
      pos= get(hc(hi), 'position');
      set(hc(hi), 'position', posShift - (posShift-pos) .* posShrink);
    end  
  end
end

ax= getBackgroundAxis;
prop= {'visible', 'on', ...
       'horizontalAlignment', 'center', ...
       'verticalAlignment', 'middle'};
if VERTICAL,
  prop= {prop{:}, 'rotation', 90};
  h= textFit(borderSize/2, 0.5, tit, [borderSize 0.9], prop{:});
else
  h= textFit(0.5, 1-borderSize/2, tit, [0.8 borderSize], prop{:});
end
