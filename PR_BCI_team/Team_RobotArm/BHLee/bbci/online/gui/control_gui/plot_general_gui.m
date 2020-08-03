function plot_general_gui(fig);
%PLOT_GENERAL_GUI ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% plots the general setup part of the gui
% 
% usage:
%    plot_general_gui(fig);
%
% input:
%    fig   the handle of the gui
%
% Guido Dornhege
% $Id: plot_general_gui.m,v 1.3 2006/06/12 14:17:41 neuro_cvs Exp $

global nice_gui_font general_port_fields
global VP_CODE

% GET THE ACTUAL SETUP
setup= control_gui_queue(fig,'get_setup');
setup.general= ...
    set_defaults(setup.general, ...
                 'player',length(general_port_fields), ...
                 'graphic',1, ...
                 'graphic_master',0, ...
                 'setup_list1',{}, ...
                 'setup_list2',{});
setup.general= ...
    set_defaults(setup.general, ...
                 'setup_list_default1',setup.general.setup_list1, ...
                 'setup_list_default2',setup.general.setup_list2, ...
                 'active1',1, ...
                 'active2',1, ...
                 'save',0, ...
                 'savestring', 'imag_fbarrow', ...
                 'stopmarker',254);
%setup.savemode = 0;
setup= set_defaults(setup, ...
                    'savemode', 0);
control_gui_queue(fig,'set_setup',setup);

% CREATE ALL BUTTONS
ax.player_text = uicontrol('Style','text','units','normalized','position',[0.05 0.9 0.25 0.04],'String','Number of Players');
set(ax.player_text,'Tooltipstring','Specify the number of players');
set(ax.player_text,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.player_text,'UserData',fig);

ax.player = uicontrol('Style','popupmenu','units','normalized','position',[0.35 0.9 0.1 0.04],'String',{'1','2'},'Value',setup.general.player);
set(ax.player,'Callback','change_taps(get(gcbo,''UserData''),''player'',get(gcbo,''Value''))');
set(ax.player,'Tooltipstring','Specify the number of players');
set(ax.player,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.player,'UserData',fig);

ax.graphic = uicontrol('Style','checkbox','units','normalized','position',[0.75 0.9 0.2 0.04],'String','Enable graphic control','Value',setup.general.graphic);
set(ax.graphic,'Callback','change_taps(get(gcbo,''UserData''),''graphic'',get(gcbo,''Value''))');
set(ax.graphic,'Tooltipstring','Should the graphic be controlled??');
set(ax.graphic,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.graphic,'UserData',fig);

ax.graphic_master = uicontrol('Style','checkbox','units','normalized','position',[0.5 0.9 0.2 0.04],'String','Enable graphic master','Value',setup.general.graphic_master);
set(ax.graphic_master,'Callback','change_taps(get(gcbo,''UserData''),''graphic_master'',get(gcbo,''Value''))');
set(ax.graphic_master,'Tooltipstring','Should the graphic master installed??');
set(ax.graphic_master,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.graphic_master,'UserData',fig);


ax.load_default = uicontrol('Style','pushbutton','units','normalized','position',[0 0 0.2 0.05],'String','Load Default');
set(ax.load_default,'Tooltipstring','Cancels the old setup, loads new one and use the default values');
set(ax.load_default,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.load_default,'Callback','load_control_setup(get(gcbo,''UserData''),''default'',''all'');');
set(ax.load_default,'UserData',fig);

ax.load_merge = uicontrol('Style','pushbutton','units','normalized','position',[0 0.05 0.2 0.05],'String','Load Merge');
set(ax.load_merge,'Tooltipstring','Loads a new setup, use the fields there but with the values of the old setup if they exist. Otherwise take the new one.');
set(ax.load_merge,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.load_merge,'Callback','load_control_setup(get(gcbo,''UserData''),''merge'',''all'');');
set(ax.load_merge,'UserData',fig);

ax.load_add = uicontrol('Style','pushbutton','units','normalized','position',[0.2 0.05 0.2 0.05],'String','Load Add');
set(ax.load_add,'Tooltipstring','Loads a new setup, and add the new fields to the existing one from the old setup.');
set(ax.load_add,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.load_add,'Callback','load_control_setup(get(gcbo,''UserData''),''add'',''all'');');
set(ax.load_add,'UserData',fig);

ax.save = uicontrol('Style','pushbutton','units','normalized','position',[0.2 0 0.2 0.05],'String','Save');
set(ax.save,'Tooltipstring','Save a setup');
set(ax.save,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.save,'Callback','save_control_setup(get(gcbo,''UserData''),''all'');');
set(ax.save,'UserData',fig);

ax.exit = uicontrol('Style','pushbutton','units','normalized','position',[0.8 0 0.2 0.05],'String','Exit');
set(ax.exit,'Tooltipstring','Exit this gui');
set(ax.exit,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.exit,'Callback','exit_control_setup(get(gcbo,''UserData''));');
set(ax.exit,'UserData',fig);


ax.stop = uicontrol('Style','pushbutton','units','normalized','position',[0.8 0.05 0.2 0.05],'String','Quit');
set(ax.stop,'Tooltipstring','Terminate all applications to this gui');
set(ax.stop,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.stop,'Callback','stop_control_setup(get(gcbo,''UserData''),''all'');');
set(ax.stop,'UserData',fig);

ax.start = uicontrol('Style','pushbutton','units','normalized','position',[0.6 0.05 0.2 0.05],'String','Start');
set(ax.start,'Tooltipstring','Start all applications to this gui');
set(ax.start,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.start,'Callback','playing_game(get(gcbo,''UserData''),''all'',''play'');');
set(ax.start,'UserData',fig);

ax.pause = uicontrol('Style','pushbutton','units','normalized','position',[0.6 0 0.2 0.05],'String','Pause');
set(ax.pause,'Tooltipstring','Pause all applications to this gui');
set(ax.pause,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.pause,'Callback','playing_game(get(gcbo,''UserData''),''all'',''pause'');');
set(ax.pause,'UserData',fig);


ax.send = uicontrol('Style','pushbutton','units','normalized','position',[0.4 0 0.2 0.05],'String','Send');
set(ax.send,'Tooltipstring','Send all values to the applications');
set(ax.send,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.send,'Callback','send_informations(get(gcbo,''UserData''),''all'',0);');
set(ax.send,'UserData',fig);

ax.sendinit = uicontrol('Style','pushbutton','units','normalized','position',[0.4 0.05 0.2 0.05],'String','Send+Init');
set(ax.sendinit,'Tooltipstring','Send all values to the applications and initialize a refresh');
set(ax.sendinit,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.sendinit,'Callback','send_informations(get(gcbo,''UserData''),''all'',1);');
set(ax.sendinit,'UserData',fig);

ax.setup_list_name1 = uicontrol('Style','text','units','normalized','position',[0.05 0.85 0.15 0.03],'String','Player 1:','HorizontalAlignment','left');
set(ax.setup_list_name1,'Tooltipstring','A list of available setups');
set(ax.setup_list_name1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.9);
set(ax.setup_list_name1,'UserData',fig);


ax.setup_list1 = uicontrol('Style','listbox','units','normalized','position',[0 0.64 0.8 0.2],'String',setup.general.setup_list1);
set(ax.setup_list1,'Tooltipstring','A list of available setups');
set(ax.setup_list1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.1);
set(ax.setup_list1,'UserData',fig);
set(ax.setup_list1,'Min',1,'Max',1,'Value',1);

ax.setup_listup1 = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.79 0.07 0.05],'String','^');
set(ax.setup_listup1,'Tooltipstring','Move this setup up');
set(ax.setup_listup1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',1);
set(ax.setup_listup1,'Callback','move_setup(get(gcbo,''UserData''),1,1);');
set(ax.setup_listup1,'UserData',fig);

ax.setup_listdown1 = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.64 0.07 0.05],'String','v');
set(ax.setup_listdown1,'Tooltipstring','Move this setup down');
set(ax.setup_listdown1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',1);
set(ax.setup_listdown1,'Callback','move_setup(get(gcbo,''UserData''),1,-1);');
set(ax.setup_listdown1,'UserData',fig);

ax.setup_listadd1 = uicontrol('Style','pushbutton','units','normalized','position',[0.9 0.78 0.1 0.06],'String','Add');
set(ax.setup_listadd1,'Tooltipstring','Adds a setup');
set(ax.setup_listadd1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.setup_listadd1,'Callback','add_setup(get(gcbo,''UserData''),1);');
set(ax.setup_listadd1,'UserData',fig);

ax.setup_listdel1 = uicontrol('Style','pushbutton','units','normalized','position',[0.9 0.71 0.1 0.06],'String','Del');
set(ax.setup_listdel1,'Tooltipstring','Deletes the setup');
set(ax.setup_listdel1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.setup_listdel1,'Callback','delete_setup(get(gcbo,''UserData''),1);');
set(ax.setup_listdel1,'UserData',fig);

ax.setup_listupd1 = uicontrol('Style','pushbutton','units','normalized','position',[0.9 0.64 0.1 0.06],'String','Upd');
set(ax.setup_listupd1,'Tooltipstring','Update the setup');
set(ax.setup_listupd1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.setup_listupd1,'Callback','update_setup(get(gcbo,''UserData''),1);');
set(ax.setup_listupd1,'UserData',fig);

ax.setup_active1 = uicontrol('Style','checkbox','units','normalized','position',[0.81 0.73 0.08 0.04],'String','Active','Value',setup.general.active1);
set(ax.setup_active1,'Callback','setup_active(get(gcbo,''UserData''),get(gcbo,''Value''),1);');
set(ax.setup_active1,'Tooltipstring','Should this player be used?');
set(ax.setup_active1,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.setup_active1,'UserData',fig);

ax.setup_list_name2 = uicontrol('Style','text','units','normalized','position',[0.05 0.55 0.15 0.03],'String','Player 2:','HorizontalAlignment','left');
set(ax.setup_list_name2,'Tooltipstring','A list of available setups');
set(ax.setup_list_name2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.9);
set(ax.setup_list_name2,'UserData',fig);

ax.setup_list2 = uicontrol('Style','listbox','units','normalized','position',[0 0.34 0.8 0.2],'String',setup.general.setup_list2);
set(ax.setup_list2,'Tooltipstring','A list of available setups');
set(ax.setup_list2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.1);
set(ax.setup_list2,'UserData',fig);
set(ax.setup_list2,'Min',1,'Max',1,'Value',1);

ax.setup_listup2 = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.49 0.07 0.05],'String','^');
set(ax.setup_listup2,'Tooltipstring','Move this setup up');
set(ax.setup_listup2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',1);
set(ax.setup_listup2,'Callback','move_setup(get(gcbo,''UserData''),2,1);');
set(ax.setup_listup2,'UserData',fig);

ax.setup_listdown2 = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.34 0.07 0.05],'String','v');
set(ax.setup_listdown2,'Tooltipstring','Move this setup down');
set(ax.setup_listdown2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',1);
set(ax.setup_listdown2,'Callback','move_setup(get(gcbo,''UserData''),2,-1);');
set(ax.setup_listdown2,'UserData',fig);

ax.setup_listadd2 = uicontrol('Style','pushbutton','units','normalized','position',[0.9 0.48 0.1 0.06],'String','Add');
set(ax.setup_listadd2,'Tooltipstring','Adds a setup');
set(ax.setup_listadd2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.setup_listadd2,'Callback','add_setup(get(gcbo,''UserData''),2);');
set(ax.setup_listadd2,'UserData',fig);

ax.setup_listdel2 = uicontrol('Style','pushbutton','units','normalized','position',[0.9 0.41 0.1 0.06],'String','Del');
set(ax.setup_listdel2,'Tooltipstring','Deletes the setup');
set(ax.setup_listdel2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.setup_listdel2,'Callback','delete_setup(get(gcbo,''UserData''),2);');
set(ax.setup_listdel2,'UserData',fig);

ax.setup_listupd2 = uicontrol('Style','pushbutton','units','normalized','position',[0.9 0.34 0.1 0.06],'String','Upd');
set(ax.setup_listupd2,'Tooltipstring','Update the setup');
set(ax.setup_listupd2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.setup_listupd2,'Callback','update_setup(get(gcbo,''UserData''),2);');
set(ax.setup_listupd2,'UserData',fig);

ax.setup_active2 = uicontrol('Style','checkbox','units','normalized','position',[0.81 0.43 0.08 0.04],'String','Active','Value',setup.general.active2);
set(ax.setup_active2,'Callback','setup_active(get(gcbo,''UserData''),get(gcbo,''Value''),2);');
set(ax.setup_active2,'Tooltipstring','Should this player be used?');
set(ax.setup_active2,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.setup_active2,'UserData',fig);

ax.imlucky = uicontrol('Style','pushbutton','units','normalized','position',[0.82 0.57 0.15 0.05],'String','I''m lucky');
set(ax.imlucky,'Tooltipstring','try to find setup files');
set(ax.imlucky,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.imlucky,'Callback','im_lucky(get(gcbo,''UserData''));');
set(ax.imlucky,'UserData',fig);

%% Checkbox button: Auto Saving On/Off
ax.savefile = uicontrol('Style','checkbox', ...
                        'units','normalized', ...
                        'position',[0.02 0.25, 0.15,0.06], ...
                        'Value',setup.general.save, ...
                        'String','save');
% claudia
% set(ax.savefile,'Tooltipstring','should the next run be saved. Note: this modus is only allowed if BrainVisionRecorder and this gui run on the same machine and BrainVisionRecorder is in monitoring mode. Then Recording and feedback are automatically started after re-initializing in this GUI.');
set(ax.savefile,'Tooltipstring','should the next run be saved? Note: this modus is only allowed if BrainVisionRecorder is in monitoring mode. Then Recording and feedback are automatically started after re-initializing in this GUI.');
set(ax.savefile,'FontUnits','normalized', ...
                'FontName',nice_gui_font, ...
                'FontSize',0.6);
set(ax.savefile,'Callback', ...
       'setup_save(get(gcbo,''UserData''),get(gcbo,''Value''));');
set(ax.savefile,'UserData',fig);

%% Text field for file name
ax.savestring = uicontrol('Style','edit', ...
                          'units','normalized', ...
                          'position',[0.2 0.25, 0.6, 0.06], ...
                          'String',setup.general.savestring);
set(ax.savestring,'Tooltipstring','name of the save string');
set(ax.savestring,'FontUnits','normalized', ...
                  'FontName',nice_gui_font, ...
                  'FontSize',0.6);
set(ax.savestring,'Callback', ...
       'setup_savestring(get(gcbo,''UserData''),get(gcbo,''String''));');
set(ax.savestring,'UserData',fig);
if setup.general.save
  set(ax.savestring,'Enable','on');
else
  set(ax.savestring,'Enable','off');
end


ax.stop_markertext = uicontrol('Style','text','units','normalized','position',[0.02 0.15, 0.16,0.06],'String','Stop Marker');
set(ax.stop_markertext,'Tooltipstring','specify stop marker');
set(ax.stop_markertext,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.stop_markertext,'UserData',fig);

ax.stop_marker = uicontrol('Style','edit','units','normalized','position',[0.2 0.15, 0.16,0.06],'String',setup.general.stopmarker);
set(ax.stop_marker,'Tooltipstring','specify stop marker');
set(ax.stop_marker,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.6);
set(ax.stop_marker,'UserData',fig);
set(ax.stop_marker,'Callback','setup_stopmarker(get(gcbo,''UserData''),get(gcbo,''String''));');
if setup.general.save
  set(ax.stop_marker,'Enable','on');
else
  set(ax.stop_marker,'Enable','off');
end

ax.interrupt = uicontrol('Style','pushbutton','units','normalized','position',[0.02 0.15, 0.8,0.15],'String','Interrupt');
set(ax.interrupt,'Tooltipstring','interrupt current saving');
set(ax.interrupt,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.3);
set(ax.interrupt,'UserData',fig);
set(ax.interrupt,'Callback','interrupt_saving(get(gcbo,''UserData''));');
set(ax.interrupt,'Visible','off');

% SAVE THE BUTTONS
control_gui_queue(fig,'set_general_ax',ax);


change_taps(fig,'player',setup.general.player);
change_taps(fig,'graphic',setup.general.graphic);

