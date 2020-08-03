function setLastXTickLabel(str, varargin)
%setLastXTickLabel(str, <forceAtLim>)
%
% set the last xTickLabel to the given string.

if nargin==2
  forceAtLim = varargin{1};
end

h= gca;
xLim= get(h, 'XLim');
xTick= get(h, 'XTick');
xTickLabel= cellstr(get(h, 'XTickLabel'));
if nargin>1 && forceAtLim==2 && xLim(2)==xTick(end),
  forceAtLim=1;
end
if nargin>1 && forceAtLim==2,
  xTickLabel{end+1}= str;
else
  xTickLabel{end}= str;
end
set(h, 'XTickMode','manual', 'XTickLabel',xTickLabel);

%% xTickMode has to be set to 'manual'. Otherwise problems will occur
%% after resizing the figure (and probably when printing it).

if nargin>1 && forceAtLim,
  if forceAtLim==2,
    xTick(end+1)= xLim(2);
  else
    xTick(end)= xLim(2);
  end
  set(h, 'XTick',xTick, 'XLimMode','manual');
end
