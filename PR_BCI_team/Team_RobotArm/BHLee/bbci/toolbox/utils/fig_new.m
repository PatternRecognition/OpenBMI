function fn= fig_new(fn)

if isempty(get(0, 'Children')),
  %% No figure open yet: Specify position of first figure
  fig_toolsoff;
  set(gcf, 'Position', [6 596 950 580]);     %% This is to please me (BB)
  return;
end
%% Otherwise take position of current figure
pos= get(gcf, 'Position');
mb= get(gcf, 'MenuBar');
tb= get(gcf, 'ToolBar');
if nargin<1,
  fn= figure;
else
  figure(fn);
end
set(fn, 'Position',pos, 'MenuBar',mb, 'ToolBar',tb);
