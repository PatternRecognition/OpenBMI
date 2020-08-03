function setup_active(fig,val,player);

setup = control_gui_queue(fig,'get_setup');

if player==1
  setup.general.active1 = val;
else
  setup.general.active2 = val;
end

control_gui_queue(fig,'set_setup',setup);

ax = control_gui_queue(fig,'get_general_ax');

if setup.general.active1
  str = 'on';
else
  str = 'off';
end

set(ax.setup_list_name1,'Enable',str);
set(ax.setup_list1,'Enable',str);
set(ax.setup_listup1,'Enable',str);
set(ax.setup_listdown1,'Enable',str);
set(ax.setup_listadd1,'Enable',str);
set(ax.setup_listdel1,'Enable',str);
set(ax.setup_listupd1,'Enable',str);



control_player1 = control_gui_queue(fig,'get_control_player1');
set(control_player1,'Enable',str);

graphic_player1 = control_gui_queue(fig,'get_graphic_player1');
set(graphic_player1,'Enable',str);

if setup.general.active2
  str = 'on';
else
  str = 'off';
end

set(ax.setup_list_name2,'Enable',str);
set(ax.setup_list2,'Enable',str);
set(ax.setup_listup2,'Enable',str);
set(ax.setup_listdown2,'Enable',str);
set(ax.setup_listadd2,'Enable',str);
set(ax.setup_listdel2,'Enable',str);
set(ax.setup_listupd2,'Enable',str);



control_player2 = control_gui_queue(fig,'get_control_player2');
set(control_player2,'Enable',str);

graphic_player2 = control_gui_queue(fig,'get_graphic_player2');
set(graphic_player2,'Enable',str);




