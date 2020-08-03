function unixLim= unifyXLim(h)
%unifyXLim(<hAxes>)
%
% unifyXLim changes the XLim setting in axes handled by "hAxes", 
% or (default) in all children of the current figure.
%

% Benjamin Blankertz, GMD-FIRST, 08/00

if ~exist('h','var'),
  h= get(gcf, 'children');
end

for hi= 1:length(h),
  isaxes(hi)= strcmp(get(h(hi), 'type'), 'axes') & ...
      ~strcmp(get(h(hi), 'tag'), 'legend');
end
h= h(find(isaxes));

for hi= 1:length(h), 
  xLim(hi,:)= get(h(hi), 'xLim'); 
end
unixLim= [min(xLim(:,1)) max(xLim(:,2))];

for hi= 1:length(h), 
  set(h(hi), 'xLim',unixLim); 
end
