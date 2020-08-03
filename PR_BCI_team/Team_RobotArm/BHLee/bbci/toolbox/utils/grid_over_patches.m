function grid_over_patches(varargin)
%grid_opver_patches
%
% replots x- and y-grid such that it is shown on top of patches

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'axes', gca, ...
                  'xGrid', [], ...
                  'yGrid', []);

for ax= opt.axes(:)',
  gridLineStyle= get(ax, 'GridLineStyle');
  if strcmp(opt.xGrid,'on') | ...
        (isempty(opt.xGrid) & strcmp(get(ax,'XGrid'), 'on')),
    xTick= get(ax, 'xTick');
    backaxes(ax);
    h= line(repmat(xTick,[2 1]), ylim');
    set(h, 'color','k', 'lineStyle',gridLineStyle, 'handleVisibility','off');
    set(ax, 'XGrid','off', 'XLimMode','Manual', 'XTickMode','Manual');
  end
  if strcmp(opt.yGrid,'on') | ...
        (isempty(opt.yGrid) & strcmp(get(ax,'YGrid'), 'on')),
    yTick= get(ax, 'yTick');
    backaxes(ax);
    h= line(xlim', repmat(yTick,[2 1]));
    set(h, 'color','k', 'lineStyle',gridLineStyle, 'handleVisibility','off');
    set(ax, 'YGrid','off', 'YLimMode','Manual', 'YTickMode','Manual');
  end
  axis_redrawFrame(ax);
end
