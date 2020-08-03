function setup_save(fig,val);

global general_port_fields

ax = control_gui_queue(fig,'get_general_ax');
set(ax.savefile,'Value',false);
drawnow;

setup = control_gui_queue(fig,'get_setup');
setup.general.save = val;
setup.savemode = val;


if setup.general.save
  str = 'on';
else
  str = 'off';
end

if setup.general.save
  host = general_port_fields.bvmachine;
  try
    bvv = acquire_bv(100,host);
    acquire_bv('close');
  catch
    bvv = [];
  end
  
  if isempty(bvv)
      setup.general.save = false;
      str = 'off';
  end   
end

set(ax.savestring,'Enable',str);
set(ax.stop_marker,'Enable',str);
set(ax.savefile,'Value',setup.general.save);

control_gui_queue(fig,'set_setup',setup);
