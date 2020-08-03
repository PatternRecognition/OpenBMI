function setLastYTickLabel(str, forceAtLim)
%setLastYTickLabel(str, <forceAtLim>)
%
% set the last yTickLabel to the given string.

h= gca;

if nargin>1 && forceAtLim,
  yLim= get(h, 'YLim');
  yTick= get(h, 'YTick');
  if yLim(2)-yTick(end) >= 0.5*mean(diff(yTick)),
    yTick(end+1)= yLim(2);
  else
    yTick(end)= yLim(2);
  end
  set(h, 'YTick',yTick, 'YLimMode','manual');
end

yTickLabel= cellstr(get(h, 'YTickLabel'));
yTickLabel{end}= str;
set(h, 'YTickMode','manual', 'YTickLabel',yTickLabel);

%% yTickMode has to be set to 'manual'. Otherwise problems will occur
%% after resizing the figure (and probably when printing it).
