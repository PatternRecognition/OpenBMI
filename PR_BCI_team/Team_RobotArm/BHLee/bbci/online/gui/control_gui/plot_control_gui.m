function plot_control_gui(fig,player,typ);
% PLOT_CONTROL_GUI ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% plots the control part of the gui
%
% usage:
%    plot_control_gui(fig,player,typ);
% 
% input:
%    fig      the handle of the gui
%    player   the player number
%    typ      control[default] or graphic
%   
% Guido Dornhege
% $Id: plot_control_gui.m,v 1.6 2007/03/29 08:05:34 neuro_cvs Exp $

global nice_gui_font general_port_fields

%some defaults
if ~exist('typ','var')
  typ = 'control';
end

setup = control_gui_queue(fig,'get_setup');

if isnumeric(player)
  if ~isfield(setup,[typ '_player' int2str(player)])
    if ~isfield(setup,[typ '_player'])
      eval(sprintf('setup.%s_player%d = struct;',typ,player));
    else
      eval(sprintf('setup.%s_player%d = setup.%s_player;',typ,player,typ));
    end
  end
  eval(sprintf('setu = setup.%s_player%d;',typ,player));
else
  if ~isfield(setup,'graphic_master')
    setup.graphic_master = struct;
  end
  setu = setup.graphic_master;
end



if isnumeric(player)
  switch typ
   case 'control'
    setu = set_defaults(setu,'machine',general_port_fields(min(length(general_port_fields),player)).control{1},'port',general_port_fields(min(length(general_port_fields),player)).control{2},'update_port',[]);
   case 'graphic'
    setu = set_defaults(setu,'machine',general_port_fields(min(length(general_port_fields),player)).graphic{1},'port',general_port_fields(min(length(general_port_fields),player)).graphic{2},'fb_port',general_port_fields(min(length(general_port_fields),player)).control{3},'update_port',[]);
  end
else
    setu = set_defaults(setu,'machine','brainamp','port',12470,'feedback_opt',struct,'update_port',[]);
    setu.feedback_opt = set_defaults(setu.feedback_opt,'client_machines',{},'client_ports',12450,'client_player',1);
end  

if ~isfield(setu,'fields')
  fi = fieldnames(setu);
  fi = setdiff(fi,{'machine','port','fb_port','fields','fields_help','update_port'});
  setu.fields = {};
  for i = 1:length(fi);
    setu.fields = cat(2,setu.fields,get_str_names(getfield(setu,fi{i}),fi{i}));
  end
end

if ~isfield(setu,'fields_help')
  setu.fields_help = cell(1,length(setu.fields));
end

if length(setu.fields_help) ~= length(setu.fields)
  error('Length does not match');
end

if isnumeric(player)
  player = sprintf('player%d',player);
end

% PLOT all navigation elements
ax.machine_text = uicontrol('Style','text','units','normalized','position',[0.01 0.9 0.12 0.04],'String','Machine');
set(ax.machine_text,'Tooltipstring','Specify the name of the machine informations should be sent to');
set(ax.machine_text,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.machine_text,'UserData',fig);

ax.machine = uicontrol('Style','edit','units','normalized','position',[0.14 0.9 0.11 0.04],'String',setu.machine);
set(ax.machine,'Callback',sprintf('setup=control_gui_queue(get(gcbo,''UserData''),''get_setup'');setup.%s_%s.machine = get(gcbo,''String'');control_gui_queue(get(gcbo,''UserData''),''set_setup'',setup);',typ,player));

set(ax.machine,'Tooltipstring','Specify the name of the machine informations should be sent to');
set(ax.machine,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.machine,'UserData',fig);

if ~isempty(strmatch('player',player))
  ax.update_port_text = uicontrol('Style','text','units','normalized','position',[0.26 0.9 0.12 0.04],'String','Update Port');
  set(ax.update_port_text,'Tooltipstring','Port Number the GUI listen to to get updates on some values');
  set(ax.update_port_text,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.update_port_text,'UserData',fig);

  ax.update_port = uicontrol('Style','edit','units','normalized','position',[0.39 0.9 0.1 0.04],'String',setu.update_port);
  set(ax.update_port,'Callback',sprintf('update_timer(get(gcbo,''UserData''),get(gcbo,''String''),''%s'',''%s'');setup=control_gui_queue(get(gcbo,''UserData''),''get_setup'');ax = control_gui_queue(get(gcbo,''UserData''),''get_%s_%s_ax'');set(ax.update_port,''String'',get_text_string(setup.%s_%s.update_port));',typ,player,typ,player,typ,player));
  %val = eval('setup.%s_%s.update_port',typ, player);
  %fprintf('Update timer started, Listening to port %d.',val);% this
  %doesn't give the right results!
  set(ax.update_port,'Tooltipstring','Port Number the GUI listen to to get updates on some values');
  set(ax.update_port,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.update_port,'UserData',fig);
end

ax.port_text = uicontrol('Style','text','units','normalized','position',[0.51 0.9 0.12 0.04],'String','Control Port');
set(ax.port_text,'Tooltipstring','Specify the port number informations should be sent to');
set(ax.port_text,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.port_text,'UserData',fig);

ax.port = uicontrol('Style','edit','units','normalized','position',[0.64 0.9 0.1 0.04],'String',num2str(setu.port));
set(ax.port,'Callback',sprintf('setup=control_gui_queue(get(gcbo,''UserData''),''get_setup'');setup.%s_%s.port = str2num(get(gcbo,''String''));control_gui_queue(get(gcbo,''UserData''),''set_setup'',setup);',typ,player));
set(ax.port,'Tooltipstring','Specify the port number informations should be sent to');
set(ax.port,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.port,'UserData',fig);

if ~strcmp(player,'master')
  ax.fb_port_text = uicontrol('Style','text','units','normalized','position',[0.76 0.9 0.12 0.04],'String','Feedback Port');
  set(ax.fb_port_text,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fb_port_text,'UserData',fig);
  set(ax.fb_port_text,'Tooltipstring','Specify the port number informations should be sent to');
  
  if strcmp(typ,'graphic') 
    ax.fb_port = uicontrol('Style','edit','units','normalized','position',[0.89 0.9 0.1 0.04],'String',num2str(setu.fb_port));
    
    set(ax.fb_port,'Callback',sprintf('setup=control_gui_queue(get(gcbo,''UserData''),''get_setup'');setup.%s_%s.fb_port = str2num(get(gcbo,''String''));control_gui_queue(get(gcbo,''UserData''),''set_setup'',setup);control_%s_ax = control_gui_queue(get(gcbo,''UserData''),''get_control_%s_ax'');set(control_%s_ax.fb_port,''String'',setup.%s_%s.fb_port);',typ,player,player,player,player,typ,player));
    set(ax.fb_port,'Tooltipstring','Specify the port number informations should be sent to');
  else
    eval(sprintf('setg = setup.graphic_%s;',player));
    
  ax.fb_port = uicontrol('Style','text','units','normalized','position',[0.89 0.9 0.1 0.04],'String',num2str(setg.fb_port));
  set(ax.fb_port,'Tooltipstring','see graphic');
  
  end

  set(ax.fb_port,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fb_port,'UserData',fig);
end

ax.load_default = uicontrol('Style','pushbutton','units','normalized','position',[0 0 0.2 0.05],'String','Load Default');
set(ax.load_default,'Tooltipstring','For the control of this player: Cancels the old setup, loads new one and use the default values');
set(ax.load_default,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.load_default,'Callback',sprintf('load_control_setup(get(gcbo,''UserData''),''default'',''%s_%s'');',typ,player));
set(ax.load_default,'UserData',fig);

ax.load_merge = uicontrol('Style','pushbutton','units','normalized','position',[0 0.05 0.2 0.05],'String','Load Merge');
set(ax.load_merge,'Tooltipstring','For the control of this player: Loads a new setup, use the fields there but with the values of the old setup if they exist. Otherwise take the new one.');
set(ax.load_merge,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.load_merge,'Callback',sprintf('load_control_setup(get(gcbo,''UserData''),''merge'',''%s_%s'');',typ,player));
set(ax.load_merge,'UserData',fig);

ax.load_add = uicontrol('Style','pushbutton','units','normalized','position',[0.2 0.05 0.2 0.05],'String','Load Add');
set(ax.load_add,'Tooltipstring','For the control of this player: Loads a new setup, and add the new fields to the existing one from the old setup.');
set(ax.load_add,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.load_add,'Callback',sprintf('load_control_setup(get(gcbo,''UserData''),''add'',''%s_%s'');',typ,player));
set(ax.load_add,'UserData',fig);

ax.save = uicontrol('Style','pushbutton','units','normalized','position',[0.2 0 0.2 0.05],'String','Save');
set(ax.save,'Tooltipstring','For the control of this player: Save a setup');
set(ax.save,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.save,'Callback',sprintf('save_control_setup(get(gcbo,''UserData''),''%s_%s'');',typ,player));
set(ax.save,'UserData',fig);

ax.exit = uicontrol('Style','pushbutton','units','normalized','position',[0.8 0 0.2 0.05],'String','Exit');
set(ax.exit,'Tooltipstring','Exit this gui');
set(ax.exit,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.exit,'Callback','exit_control_setup(get(gcbo,''UserData''));');
set(ax.exit,'UserData',fig);


ax.stop = uicontrol('Style','pushbutton','units','normalized','position',[0.8 0.05 0.2 0.05],'String','Quit');
set(ax.stop,'Tooltipstring','For the control of this player: Terminate the application');
set(ax.stop,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.stop,'Callback',sprintf('stop_control_setup(get(gcbo,''UserData''),''%s_%s'');',typ,player));
set(ax.stop,'UserData',fig);

ax.start = uicontrol('Style','pushbutton','units','normalized','position',[0.6 0.05 0.2 0.05],'String','Start');
set(ax.start,'Tooltipstring','For the control of this player: Start this application');
set(ax.start,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.start,'Callback',sprintf('playing_game(get(gcbo,''UserData''),''%s_%s'',''play'');',typ,player));
set(ax.start,'UserData',fig);

ax.pause = uicontrol('Style','pushbutton','units','normalized','position',[0.6 0 0.2 0.05],'String','Pause');
set(ax.pause,'Tooltipstring','For the control of this player: Pause this application');
set(ax.pause,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.pause,'Callback',sprintf('playing_game(get(gcbo,''UserData''),''%s_%s'',''pause'');',typ,player));
set(ax.pause,'UserData',fig);


ax.send = uicontrol('Style','pushbutton','units','normalized','position',[0.4 0 0.2 0.05],'String','Send');
set(ax.send,'Tooltipstring','Send all control values for this player to the applications');
set(ax.send,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.send,'Callback',sprintf('send_informations(get(gcbo,''UserData''),''%s_%s'',0);',typ,player));
set(ax.send,'UserData',fig);

ax.sendinit = uicontrol('Style','pushbutton','units','normalized','position',[0.4 0.05 0.2 0.05],'String','Send+Init');
set(ax.sendinit,'Tooltipstring','Send all control values for this player to the applications and initialize a refresh');
set(ax.sendinit,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ax.sendinit,'Callback',sprintf('send_informations(get(gcbo,''UserData''),''%s_%s'',1);',typ,player));
set(ax.sendinit,'UserData',fig);

ax.fields = zeros(length(setu.fields),8);

if length(setu.fields)<16
  begin = 0.85; width = 0.05;
else
  begin = 0.9; width = 0.8/length(setu.fields);
end

for i = 1:length(setu.fields)
  ax.fields(i,1) = uicontrol('Style','text','units','normalized','position',[0.01 begin-i*width 0.25 width*0.9]);
  switch typ
   case 'control'
    if ~isempty(strmatch('bbci.',setu.fields{i}))
      str = setu.fields{i}(5:end);
    else
      str = setu.fields{i};
    end
   case 'graphic'
    if ~isempty(strmatch('feedback_opt.',setu.fields{i}))
      str = setu.fields{i}(13:end);
    else
      str = setu.fields{i};
    end
  end
  
  set(ax.fields(i,1),'String',str);
  if ~isempty(setu.fields_help{i})
    set(ax.fields(i,1),'Tooltipstring',setu.fields_help{i});
  end
  
  set(ax.fields(i,1),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,1),'UserData',fig,'HorizontalAlignment','left');
  
  ax.fields(i,2) = uicontrol('Style','edit','units','normalized','position',[0.3 begin-i*width 0.4 width*0.9]);
  str = eval(sprintf('setu.%s',setu.fields{i}));
  str = get_text_string(str);
  
  set(ax.fields(i,2),'String',str);
  if ~isempty(setu.fields_help{i})
    set(ax.fields(i,2),'Tooltipstring',setu.fields_help{i});
  end
  
  set(ax.fields(i,2),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,2),'UserData',fig);
  set(ax.fields(i,2),'Callback',sprintf('activate_this_entry(get(gcbo,''UserData''),''setup.%s_%s.%s'',get(gcbo,''String''));',typ,player,setu.fields{i}));
  
  ax.fields(i,3) = uicontrol('Style','pushbutton','units','normalized','position',[0.75 begin-i*width 0.048 width*0.9],'String','S');
  
  set(ax.fields(i,3),'Tooltipstring','Send this entry');
  

  
  set(ax.fields(i,3),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,3),'UserData',fig);
  set(ax.fields(i,3),'Callback',sprintf('send_informations(get(gcbo,''UserData''),''%s_%s'',''%s'');',typ,player,setu.fields{i}));
  
  
   ax.fields(i,4) = uicontrol('Style','pushbutton','units','normalized','position',[0.8 begin-i*width 0.048 width*0.9],'String','M');
  
  set(ax.fields(i,4),'Tooltipstring','Modify this entry');
  

  
  set(ax.fields(i,4),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,4),'UserData',fig);
  set(ax.fields(i,4),'Callback',sprintf('modify_field(get(gcbo,''UserData''),''%s_%s'',''%s'');',typ,player,setu.fields{i}));
  
  
   ax.fields(i,5) = uicontrol('Style','pushbutton','units','normalized','position',[0.85 begin-i*width 0.048 width*0.9],'String','A');
  
  set(ax.fields(i,5),'Tooltipstring','Add an entry behind this point');
  

  
  set(ax.fields(i,5),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,5),'UserData',fig);
  set(ax.fields(i,5),'Callback',sprintf('add_entry(get(gcbo,''UserData''),''%s_%s'',''%s'');',typ,player,setu.fields{i}));
  
   ax.fields(i,6) = uicontrol('Style','pushbutton','units','normalized','position',[0.9 begin-i*width 0.048 width*0.44],'String','v');
  
  set(ax.fields(i,6),'Tooltipstring','Move this entry down');
  
  
  set(ax.fields(i,6),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,6),'UserData',fig);
  set(ax.fields(i,6),'Callback',sprintf('move_down(get(gcbo,''UserData''),''%s_%s'',''%s'');',typ,player,setu.fields{i}));
  
  
   ax.fields(i,7) = uicontrol('Style','pushbutton','units','normalized','position',[0.9 begin-i*width+0.46*width 0.048 width*0.44],'String','^');
  
  set(ax.fields(i,7),'Tooltipstring','Move this entry up');
  
  
  set(ax.fields(i,7),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,7),'UserData',fig);
  set(ax.fields(i,7),'Callback',sprintf('move_up(get(gcbo,''UserData''),''%s_%s'',''%s'');',typ,player,setu.fields{i}));
  
  
   ax.fields(i,8) = uicontrol('Style','pushbutton','units','normalized','position',[0.95 begin-i*width 0.048 width*0.9],'String','D');
  
  set(ax.fields(i,8),'Tooltipstring','Delete this entry');
  
  
  set(ax.fields(i,8),'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields(i,8),'UserData',fig);
  set(ax.fields(i,8),'Callback',sprintf('delete_entry(get(gcbo,''UserData''),''%s_%s'',''%s'');',typ,player,setu.fields{i}));
  
  
end

if length(setu.fields)==0
  ax.fields = uicontrol('Style','pushbutton','units','normalized','position',[0.3 0.3 0.4 0.2],'String','Add');
  
  set(ax.fields,'Tooltipstring','Add the first entry');
  
  
  set(ax.fields,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
  set(ax.fields,'UserData',fig);
  set(ax.fields,'Callback',sprintf('add_entry(get(gcbo,''UserData''),''%s_%s'',[]);',typ,player));
end

if ~isempty(setu.update_port)
  update_timer(fig,setu.update_port,typ,player);
end

% SAVE IT ALL
setup.gui_machine = get_hostname;
setu.gui_machine = get_hostname;
eval(sprintf('setup.%s_%s = setu;',typ,player));
control_gui_queue(fig,'set_setup',setup);

control_gui_queue(fig,['set_' typ '_' player '_ax'],ax);


return;


function str2 = get_str_names(var,str);
% GET ALL FIELDS OF VAR AS STRING
if isstruct(var)
  fi = fieldnames(var);
  str2 = {};
  for i = 1:length(fi);
    str2 = cat(2,str2,get_str_names(getfield(var,fi{i}),[str '.' fi{i}]));
  end
else
  str2 = {str};
end

  