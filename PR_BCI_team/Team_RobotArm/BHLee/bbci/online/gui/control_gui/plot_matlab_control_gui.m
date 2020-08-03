function plot_matlab_control_gui(fig);
%PLOT_MATLAB_CONTROL_GUI ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% plot the general framework of the gui and call the specific axis to be plotted
%
% usage:
%     plot_matlab_control_gui(fig);
%
% input:
%     fig    handle of the gui
%
% Guido Dornhege
% $Id: plot_matlab_control_gui.m,v 1.2 2007/01/11 15:19:22 neuro_cvs Exp $

global nice_gui_font

% get the setup and set defaults
setup = control_gui_queue(fig,'get_setup');
setup = set_defaults(setup,'general',struct);
setup.general = set_defaults(setup.general,'position',[100 100 800 600]);
control_gui_queue(fig,'set_setup',setup);

% create the figure
figure(fig);
clf(fig);
set(fig,'MenuBar','none','position',setup.general.position,'Color',[0.9,0.9,0.9]);
set(fig,'NumberTitle','off')
set(fig,'Name','CONTROL_GUI');
set(fig,'UserData',fig);
set(fig,'CloseRequestFcn','close_control_figure(get(gcbo,''UserData''));');

set(gca,'Units','normalized','position',[0 0 1 1]);
axis off

% build the taps
general = uicontrol('style','pushbutton','units','normalized','position',[0,0.95,0.16,0.05]);
set(general,'String','GENERAL');
set(general,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(general,'Tooltipstring','Show the general overview figure');
set(general,'Callback','activate_control_gui(get(gcbo,''UserData''),''general'');');
set(general,'UserData',fig);
control_gui_queue(fig,'set_general',general);

control_player1 = uicontrol('style','pushbutton','units','normalized','position',[0.16,0.95,0.16,0.05]);
set(control_player1,'String','CONTROL_PLAYER1');
set(control_player1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(control_player1,'Tooltipstring','Show the control_player1 overview figure');
set(control_player1,'Callback','activate_control_gui(get(gcbo,''UserData''),''control_player1'');');
set(control_player1,'UserData',fig);
control_gui_queue(fig,'set_control_player1',control_player1);

control_player2 = uicontrol('style','pushbutton','units','normalized','position',[0.32,0.95,0.16,0.05]);
set(control_player2,'String','CONTROL_PLAYER2');
set(control_player2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(control_player2,'Tooltipstring','Show the control_player2 overview figure');
set(control_player2,'Callback','activate_control_gui(get(gcbo,''UserData''),''control_player2'');');
set(control_player2,'UserData',fig);
control_gui_queue(fig,'set_control_player2',control_player2);


graphic_master = uicontrol('style','pushbutton','units','normalized','position',[0.48,0.95,0.16,0.05]);
set(graphic_master,'String','GRAPHIC_MASTER');
set(graphic_master,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(graphic_master,'Tooltipstring','Show the graphic_player_master overview figure');
set(graphic_master,'Callback','activate_control_gui(get(gcbo,''UserData''),''graphic_master'');');
set(graphic_master,'UserData',fig);
control_gui_queue(fig,'set_graphic_master',graphic_master);


graphic_player1 = uicontrol('style','pushbutton','units','normalized','position',[0.64,0.95,0.16,0.05]);
set(graphic_player1,'String','GRAPHIC_PLAYER1');
set(graphic_player1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(graphic_player1,'Tooltipstring','Show the graphic_player1 overview figure');
set(graphic_player1,'Callback','activate_control_gui(get(gcbo,''UserData''),''graphic_player1'');');
set(graphic_player1,'UserData',fig);
control_gui_queue(fig,'set_graphic_player1',graphic_player1);

graphic_player2 = uicontrol('style','pushbutton','units','normalized','position',[0.8,0.95,0.16,0.05]);
set(graphic_player2,'String','GRAPHIC_PLAYER2');
set(graphic_player2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(graphic_player2,'Tooltipstring','Show the graphic_player2 overview figure');
set(graphic_player2,'Callback','activate_control_gui(get(gcbo,''UserData''),''graphic_player2'');');
set(graphic_player2,'UserData',fig);
control_gui_queue(fig,'set_graphic_player2',graphic_player2);

% call the specific parts of the guis
plot_master_gui(fig);
plot_graphic_gui(fig,1);
plot_graphic_gui(fig,2);
plot_control_gui(fig,1);
plot_control_gui(fig,2);
plot_general_gui(fig);

