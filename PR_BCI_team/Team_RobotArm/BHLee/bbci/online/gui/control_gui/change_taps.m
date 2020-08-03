function change_taps(fig,task,val);
% CHANGE_TAPS ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% change the possible taps of the gui (2 player or graphic)
%
% usage:
%    change_taps(fig,task,val);
%
% input:
%    fig    the handle of the gui
%    task   player or graphic
%    val    the actual value of this field
%
% Guido Dornhege
% $Id: change_taps.m,v 1.2 2006/06/12 14:17:41 neuro_cvs Exp $

% get the setup and modify

setup = control_gui_queue(fig,'get_setup');

switch task
 case 'player'
  setup.general.player = val;
 case 'graphic'
  setup.general.graphic = val;
 case 'graphic_master'
  setup.general.graphic_master = val;
end

control_gui_queue(fig,'set_setup',setup);


% change the taps
general = control_gui_queue(fig,'get_general');
control_player1 = control_gui_queue(fig,'get_control_player1');
control_player2 = control_gui_queue(fig,'get_control_player2');
graphic_master = control_gui_queue(fig,'get_graphic_master');
graphic_player1 = control_gui_queue(fig,'get_graphic_player1');
graphic_player2 = control_gui_queue(fig,'get_graphic_player2');
graphic_player1_ax = control_gui_queue(fig,'get_graphic_player1_ax');
graphic_player2_ax = control_gui_queue(fig,'get_graphic_player2_ax');

if setup.general.player==1 
  if setup.general.graphic_master == 1
    set(control_player2,'Visible','off');
    set(graphic_player2,'Visible','off');
    set(graphic_master,'Visible','on');
    set(general,'Position',[0 0.95 0.25 0.05]);
    set(control_player1,'Position',[0.25 0.95 0.25 0.05]);
    set(graphic_master,'Position',[0.5 0.95 0.25,0.05]);
    set(graphic_player1,'Position',[0.75 0.95 0.25,0.05]);
  else
    set(control_player2,'Visible','off');
    set(graphic_player2,'Visible','off');
    set(graphic_master,'Visible','off');
    set(general,'Position',[0 0.95 0.33 0.05]);
    set(control_player1,'Position',[0.33 0.95 0.33 0.05]);
    set(graphic_player1,'Position',[0.67 0.95 0.33,0.05]);
  end
else
  if setup.general.graphic_master == 1
    set(graphic_player2,'Visible','on');
    set(graphic_player1,'Visible','on');
    set(control_player2,'Visible','on');
    set(graphic_master,'Visible','on');
    set(general,'Position',[0 0.95 0.166 0.05]);
    set(control_player1,'Position',[0.166 0.95 0.166 0.05]);
    set(control_player2,'Position',[0.332 0.95 0.166 0.05]);
    set(graphic_master,'Position',[0.498 0.95 0.166 0.05]);
    set(graphic_player1,'Position',[0.664 0.95 0.166 0.05]);
    set(graphic_player2,'Position',[0.83 0.95 0.166 0.05]);
  else
    set(graphic_player2,'Visible','on');
    set(graphic_player1,'Visible','on');
    set(control_player2,'Visible','on');
    set(graphic_master,'Visible','off');
    set(general,'Position',[0 0.95 0.2 0.05]);
    set(control_player1,'Position',[0.2 0.95 0.2 0.05]);
    set(control_player2,'Position',[0.4 0.95 0.2 0.05]);
    set(graphic_player1,'Position',[0.6 0.95 0.2 0.05]);
    set(graphic_player2,'Position',[0.8 0.95 0.2 0.05]);
  end
end




    

ax = control_gui_queue(fig,'get_general_ax');

if setup.general.player==1
  set(ax.setup_list2,'Visible','off');
  set(ax.setup_list_name2,'Visible','off');
  set(ax.setup_listup2,'Visible','off');
  set(ax.setup_listdown2,'Visible','off');
  set(ax.setup_listadd2,'Visible','off');
  set(ax.setup_listdel2,'Visible','off');
  set(ax.setup_listupd2,'Visible','off');
  set(ax.setup_active2,'Visible','off');
else
  set(ax.setup_list2,'Visible','on');
  set(ax.setup_list_name2,'Visible','on');
  set(ax.setup_listup2,'Visible','on');
  set(ax.setup_listdown2,'Visible','on');
  set(ax.setup_listadd2,'Visible','on');
  set(ax.setup_listdel2,'Visible','on');
  set(ax.setup_listupd2,'Visible','on');
  set(ax.setup_active2,'Visible','on');
end
