function setup_stopmarker(fig,val);

setup = control_gui_queue(fig,'get_setup');

setup.general.stopmarker = str2num(val);

control_gui_queue(fig,'set_setup',setup);
