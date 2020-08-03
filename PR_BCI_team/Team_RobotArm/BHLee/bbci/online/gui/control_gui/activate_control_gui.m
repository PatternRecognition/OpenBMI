function activate_control_gui(fig,task);
% ACTIVATE_CONTROL_GUI ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% activate the tap
% 
% usage:
%     activate_control_gui(fig,task);
%
% input:
%     fig    the handle of the gui
%     task   the tap to be activated (general,control_player1,control_player2,
%                                     graphic_player1,graphic_player2)
%
% Guido Dornhege
% $Id: activate_control_gui.m,v 1.2 2006/06/12 14:17:41 neuro_cvs Exp $

% get the axis

general_ax = control_gui_queue(fig,'get_general_ax');
control_player1_ax = control_gui_queue(fig,'get_control_player1_ax');
control_player2_ax = control_gui_queue(fig,'get_control_player2_ax');
graphic_master_ax = control_gui_queue(fig,'get_graphic_master_ax');
graphic_player1_ax = control_gui_queue(fig,'get_graphic_player1_ax');
graphic_player2_ax = control_gui_queue(fig,'get_graphic_player2_ax');
general = control_gui_queue(fig,'get_general');
control_player1 = control_gui_queue(fig,'get_control_player1');
control_player2 = control_gui_queue(fig,'get_control_player2');
graphic_master = control_gui_queue(fig,'get_graphic_master');
graphic_player1 = control_gui_queue(fig,'get_graphic_player1');
graphic_player2 = control_gui_queue(fig,'get_graphic_player2');

setup= control_gui_queue(fig,'get_setup');

% deactivate all
set(general,'FontWeight','normal', 'BackgroundColor',0.8*[1 1 1]);
set(control_player1,'FontWeight','normal', 'BackgroundColor',0.8*[1 1 1]);
set(control_player2,'FontWeight','normal', 'BackgroundColor',0.8*[1 1 1]);
set(graphic_master,'FontWeight','normal', 'BackgroundColor',0.8*[1 1 1]);
set(graphic_player1,'FontWeight','normal', 'BackgroundColor',0.8*[1 1 1]);
set(graphic_player2,'FontWeight','normal', 'BackgroundColor',0.8*[1 1 1]);
change_visibility(general_ax,'off');
change_visibility(control_player1_ax,'off');  
change_visibility(control_player2_ax,'off');  
change_visibility(graphic_master_ax,'off');  
change_visibility(graphic_player1_ax,'off');  
change_visibility(graphic_player2_ax,'off');  

setup = control_gui_queue(fig,'get_setup');

% activate the one to be activated
switch task
 case 'general'
  change_visibility(general_ax,'on');
  if setup.general.player==1
    set(general_ax.setup_list_name2,'Visible','off');
    set(general_ax.setup_list2,'Visible','off');
    set(general_ax.setup_listup2,'Visible','off');
    set(general_ax.setup_listdown2,'Visible','off');
    set(general_ax.setup_listadd2,'Visible','off');
    set(general_ax.setup_listdel2,'Visible','off');
    set(general_ax.setup_listupd2,'Visible','off');
    set(general_ax.setup_active2,'Visible','off');
  end
  if setup.savemode
    set(general_ax.savefile,'Visible','off');
    set(general_ax.savestring,'Visible','off');
    set(general_ax.stop_marker,'Visible','off');
    set(general_ax.stop_markertext,'Visible','off');
  else
    set(general_ax.interrupt,'Visible','off');
  end
  set(general,'FontWeight','bold', 'BackgroundColor',[0.6 0 0]);
  set(fig,'Name','CONTROL_GUI: general'); 
 case 'control_player1'
  change_visibility(control_player1_ax,'on');
  set(control_player1,'FontWeight','bold', 'BackgroundColor',[0.6 0 0]);
  set(fig,'Name','CONTROL_GUI: control player 1'); 
 case 'control_player2'
  change_visibility(control_player2_ax,'on');
  set(control_player2,'FontWeight','bold', 'BackgroundColor',[0.6 0 0]);
  set(fig,'Name','CONTROL_GUI: control player 2'); 
 case 'graphic_master'
  change_visibility(graphic_master_ax,'on');
  
  %TODO more???
  set(graphic_master,'FontWeight','bold', 'BackgroundColor',[0.6 0 0]);
  set(fig,'Name','CONTROL_GUI: graphic master'); 
 case 'graphic_player1'
  if setup.general.graphic
    change_visibility(graphic_player1_ax,'on');
  else
    set(graphic_player1_ax.machine_text,'Visible','on');
    set(graphic_player1_ax.machine,'Visible','on');
    set(graphic_player1_ax.fb_port_text,'Visible','on');
    set(graphic_player1_ax.fb_port,'Visible','on');
    set(graphic_player1_ax.exit,'Visible','on');

  end

  if setup.general.graphic_master
    set(graphic_player1_ax.machine,'Visible','off');
    set(graphic_player1_ax.machine_text,'Visible','off');
    set(graphic_player1_ax.port,'Visible','off');
    set(graphic_player1_ax.port_text,'Visible','off');
  end
  
  set(graphic_player1,'FontWeight','bold', 'BackgroundColor',[0.6 0 0]);
  set(fig,'Name','CONTROL_GUI: graphic player 1'); 
 case 'graphic_player2'
  if setup.general.graphic
    change_visibility(graphic_player2_ax,'on');
  else
    set(graphic_player2_ax.machine_text,'Visible','on');
    set(graphic_player2_ax.machine,'Visible','on');
    set(graphic_player2_ax.fb_port_text,'Visible','on');
    set(graphic_player2_ax.fb_port,'Visible','on');
    set(graphic_player2_ax.exit,'Visible','on');

  end
  if setup.general.graphic_master
    set(graphic_player2_ax.machine,'Visible','off');
    set(graphic_player2_ax.machine_text,'Visible','off');
    set(graphic_player2_ax.port,'Visible','off');
    set(graphic_player2_ax.port_text,'Visible','off');
  end
  set(graphic_player2,'FontWeight','bold', 'BackgroundColor',[0.6 0 0]);
  set(fig,'Name','CONTROL_GUI: graphic player 2'); 
end



function change_visibility(h,task);
% intenr function to change iteratively all visibilities
fi = fieldnames(h);

for i = 1:length(fi);
  a = getfield(h,fi{i});
  set(a,'Visible',task);
end
