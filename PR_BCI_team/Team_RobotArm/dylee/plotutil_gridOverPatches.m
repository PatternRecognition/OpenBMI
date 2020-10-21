function h=plotutil_gridOverPatches(varargin)
%plotutil_gridOverPatches
%
% replots x- and y-grid such that it is shown on top of patches


props = {'Axes',          	gca         '!GRAPHICS';
         'XGrid'            []          'CHAR(on off)';
         'YGrid'            []          'CHAR(on off)';
         };

if nargin==0,
  h= props; return
else
  h=[];
end

% This solves all the issues
set(gca, 'Layer','top');

return


%% ---- old code - NOT USED


opt= opt_proplistToStruct(varargin{:});
[opt, isdefault]= opt_setDefaults(opt, props);
opt_checkProplist(opt, props);

for ax= opt.Axes(:)',
  gridLineStyle= get(ax, 'GridLineStyle');
  if strcmp(opt.XGrid,'on') || ...
        (isempty(opt.XGrid) && strcmp(get(ax,'XGrid'), 'on')),
    xTick= get(ax, 'xTick');
    visutil_backaxes(ax);
    h= line(repmat(xTick,[2 1]), ylim');
    set(h, 'color','k', 'lineStyle',gridLineStyle, 'handleVisibility','off');
    set(ax, 'XGrid','off', 'XLimMode','Manual', 'XTickMode','Manual');
  end
  if strcmp(opt.YGrid,'on') || ...
        (isempty(opt.YGrid) && strcmp(get(ax,'YGrid'), 'on')),
    yTick= get(ax, 'yTick');
    visutil_backaxes(ax);
    h= line(xlim', repmat(yTick,[2 1]));
    set(h, 'color','k', 'lineStyle',gridLineStyle, 'handleVisibility','off');
    set(ax, 'YGrid','off', 'YLimMode','Manual', 'YTickMode','Manual');
  end
  axis_redrawFrame(ax);
end