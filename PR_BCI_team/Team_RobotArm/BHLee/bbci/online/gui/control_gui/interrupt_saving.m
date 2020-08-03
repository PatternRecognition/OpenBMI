function interrupt_saving(fig);

setup = control_gui_queue(fig,'get_setup');
% if ~setup.savemode,
if ~setup.general.save,
  return;
end

%interrupt_timer(fig);

bvr_sendcommand('stoprecording');

ax = control_gui_queue(fig,'get_general_ax');
set(ax.interrupt,'Visible','off');
set(ax.savefile,'Visible','on');
set(ax.savestring,'Visible','on');
set(ax.stop_marker,'Visible','on');
set(ax.stop_markertext,'Visible','on');

setup.savemode = false;
control_gui_queue(fig,'set_setup',setup);
