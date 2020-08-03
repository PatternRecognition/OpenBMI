function modify_field(fig,typ,str);
% MODIFY_FIELD ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% modifies the entries of one gui element
%
% usage:
%     modify_field(fig,typ,str);
%
% input: 
%     fig     the handle of the gui
%     typ     control_player1, control_player2, graphic_player1,graphic_player2
%     str     the field entry to change
%
% Guido Dornhege
% $Id: modify_field.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% gui visible off
set(fig,'Visible','off');
drawnow;

% get setup
setup = control_gui_queue(fig,'get_setup');

eval(sprintf('setu = setup.%s;',typ));

ind = strcmp(str,setu.fields);
eval(sprintf('val = setu.%s;',str));

% ask for new entries
[name,tool,value,visible] = ask_for_fields(fig,str,setu.fields_help{ind},val,1);

% change the values if it makes sense and replot
if ~isempty(name)
  if name(1)=='.'
    if isempty(strmatch('control',typ));
      name = ['feedback_opt' name];
    else
      name = ['bbci' name];
    end
  end
  setu = remove_field(setu,str);
  eval(sprintf('setu.%s = value;',name));
  if visible
    setu.fields{ind} = name;
    setu.fields_help{ind} = tool;
  else
    setu.fields(ind) = [];
    setu.fields_help(ind) = [];
  end
  
  switch typ 
   case 'control_player1'
    setup.control_player1 = setu; 
   case 'control_player2'
    setup.control_player2 = setu; 
   case 'graphic_player1'
    setup.graphic_player1 = setu; 
   case 'graphic_player2'
    setup.graphic_player2 = setu; 
  end
  control_gui_queue(fig,'set_setup',setup);
  
  switch typ
   case 'control_player1'
    plot_control_gui(fig,1);
   case 'control_player2'
    plot_control_gui(fig,2);
   case 'graphic_player1'
    plot_graphic_gui(fig,1);
   case 'graphic_player2'
    plot_graphic_gui(fig,2);
  end
end

% turn the gui on
set(fig,'Visible','on');

