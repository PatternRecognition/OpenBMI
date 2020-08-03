function setup_savestring(fig,val);

setup = control_gui_queue(fig,'get_setup');

setup.general.savestring = val;

control_gui_queue(fig,'set_setup',setup);
