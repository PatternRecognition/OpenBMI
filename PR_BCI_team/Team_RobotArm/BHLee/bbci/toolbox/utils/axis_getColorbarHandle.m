function hcb= axis_getColorbarHandle(ax)
%AXIS_GETCOLORBARHANDLE - gets the handle of colorbar(s)
%
%Synopsis:
% hcb= axis_getColorbarHandle
% hcb= axis_getColorbarHandle(AX)
%
%Arguments:
% AX: vector of axis handles
%
%Returns:
% hcb: handles of all colorbars of the current figure (if AX is not given)
%      or handle(s) of axes AX.

if nargin>0,
  fig= get(ax, 'Parent');
else
  fig= gcf;
  ax= [];
end

hcb= findobj(fig, 'Tag','Colorbar');
if ~isempty(ax),
  for ii= 1:length(hcb),
    hax= handle(hcb(ii));
    if isempty(double(hax.axes)) || ~ismember(double(hax.axes), ax),
      ud= get(hcb(ii), 'UserData');
      if ~isfield(ud, 'ParentAxis') || ~ismember(ud.ParentAxis, ax),
        hcb(ii)= NaN;
      end
    end
  end
  hcb(isnan(hcb))= [];
end

