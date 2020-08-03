function H= grid_markIval(ival, chans, markCol)
%grid_markIval(ival, <chans, markCol>)
%grid_markIval(ival, <axis_handles, markCol>)
%
% IN  ival    - interval [msec], may also contain several intervals,
%                each as one row
%     chans   - channels which should be marked, default [] meaning all
%     markCol - color of the patch, if it is scalar take it as gray value,
%                default 0.8

fig_visible = strcmp(get(gcf,'Visible'),'on'); % If figure is already invisible jvm_* functions should not be called
if fig_visible
  jvm= jvm_hideFig;
end

if ~exist('chans','var'), chans=[]; end
if ~exist('markCol','var'), markCol= 0.85; end
if length(markCol)==1,
  markCol= markCol*[1 1 1];
end

if size(ival,1)>1 & size(ival,2)==2,
  for ib= 1:size(ival,1),
    H(ib)= grid_markIval(ival(ib,:), chans, markCol);
  end
  return
end


old_ax= gca;
if isnumeric(chans) & ~isempty(chans),
  hsp= chans;
else
  hsp= grid_getSubplots(chans);
end
k= 0;
for ih= hsp,
  k= k+1;
  backaxes(ih);  %% this lets the legend vanish behind the axis
  yPatch= get(ih, 'yLim');
  H.line(:,k)= line(ival([1 2; 1 2]), yPatch([1 1; 2 2]), ...
                  'color',0.5*markCol, 'LineWidth',0.3);
  
  moveObjectBack(H.line(:,k));
  H.patch(k)= patch(ival([1 2 2 1]), yPatch([1 1 2 2]), markCol);
  moveObjectBack(H.patch(k));
  grid_over_patches;
  if ~isnan(getfield(get(ih,'UserData'), 'hleg')), %% restore legend
    legend;
  end
end
set(H.line, 'UserData','ival line');
set(H.patch, 'EdgeColor','none', 'UserData','ival patch');
backaxes(old_ax);
if isfield(get(old_ax,'UserData'),'hleg') & ...
      ~isnan(getfield(get(old_ax,'UserData'), 'hleg')),
  legend;
end

if fig_visible
  jvm_restoreFig(jvm);
end
