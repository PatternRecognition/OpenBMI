function unicLim= unifyCLim(h, unicLim)
%CLim= unifyCLim(<h>)
%
% changes the CLim setting in all axes pointed to by handles h,
% or in all children of the given figure if h is not given or empty.

% blanker@cs.tu-berlin.de, 09/2009


if ~exist('h','var') | isempty(h),
  h= get(gcf, 'children');
end
hax= findobj(h, 'Type','axes', 'Tag','');
if ~exist('unicLim','var') || isempty(unicLim)    
    for hi= 1:length(hax),
        cLim(hi,:)= get(hax(hi), 'CLim');
    end
    unicLim= [min(cLim(:,1)) max(cLim(:,2))];
end
set(hax, 'CLim',unicLim);
