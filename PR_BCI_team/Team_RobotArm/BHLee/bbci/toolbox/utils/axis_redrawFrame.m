function h= axis_redrawFrame(ax, varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'vpos', 0, ...
                 'LineWidth', 0.5);

if nargin==0,
  ax= gca;
end

old_ax= gca;

for ii= 1:length(ax),
  if isequal(get(ax(ii),'XColor'),[1 1 1]) & ...
        isequal(get(ax(ii),'YColor'),[1 1 1]),
    continue;
  end
  backaxes(ax(ii));

  xLim= get(ax(ii), 'XLim');
  yLim= get(ax(ii), 'YLim');
  location= {'top','right','bottom','left'};
  XX= [xLim; xLim([2 2]); xLim; xLim([1 1])]';
  YY= [yLim([2 2]); yLim; yLim([1 1]); yLim]';
  if strcmp(get(ax(ii),'box'), 'on'),
    draw_loc= location;
  else
    draw_loc= {get(ax(ii),'XAxisLocation'), get(ax(ii),'YAxisLocation')};
  end
  idx= find(ismember(location, draw_loc));
  hold_state= get(ax(ii), 'NextPlot');
  set(ax(ii), 'NextPlot','add');
  h= plot(XX(:,idx), YY(:,idx), 'LineWidth',opt.LineWidth);
%  set(ax(ii), 'NextPlot','replace');
  set(ax(ii), 'NextPlot',hold_state);
  set(h, 'handleVisibility','off');
  hx= h(find(ismember(location(idx), {'top','bottom'})));
  hy= h(find(ismember(location(idx), {'left','right'})));
  if ~isempty(hx),
    col= get(ax(ii),'XColor');
    if isequal(col, [1 1 1]),
      delete(hx);
      h= setdiff(h, hx);
    else
      set(hx, 'Color',col);
    end
  end
  if ~isempty(hy),
    col= get(ax(ii),'YColor');
    if isequal(col, [1 1 1]),
      delete(hy);
      h= setdiff(h, hy);
    else
      set(hy, 'Color',col);
    end
  end
%  legend;  %% restore legend
end

%if old_ax~=ax(end),  %% if this was not checked the legend would disappear
  backaxes(old_ax);
%end
