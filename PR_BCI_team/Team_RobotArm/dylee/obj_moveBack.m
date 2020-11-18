function obj_moveBack(ho)
%OBJ_MOVEBACK - Move graphic object to the background
%
%Synposis:
% obj_moveBack(<HO>)
%
%Input:
% HO: handle of graphic object, default: last plotted object

if ~exist('ho', 'var'),
  ho= gco;
elseif length(ho)>1,
  for h= ho(:)',
    obj_moveBack(h);
  end
  return;
end

hp= get(ho, 'parent');
hc= get(hp, 'children');
if ~exist('ho', 'var') || isempty(ho),
  hi= length(hc);
else
  hi= find(hc==ho);
  if isempty(hi),
    error('handle not found in current axes');
  end
end

set(hp, 'children', [hc([1:hi-1, hi+1:end, hi])]);