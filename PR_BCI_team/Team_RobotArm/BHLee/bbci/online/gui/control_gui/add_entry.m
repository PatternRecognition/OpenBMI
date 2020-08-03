function add_entry(fig,typ,str);
% ADD_ENTRY ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
% 
% adds an entry to a gui
% 
% usage:
%     add_entry(fig,typ,str);
%
% input:
%     fig   handle of the gui
%     typ   control_player1, control_player2, graphic_player1,graphic_player2
%     str   the field after which the new entry should be added
%
% Guido Dornhege
% $Id: add_entry.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% disable gui
set(fig,'Visible','off');
drawnow;

% get new fields
[name,tool,value,visible] = ask_for_fields(fig,'','',[],1);


% add new fields
if ~isempty(name)
  if name(1)=='.'
    if isempty(strmatch('control',typ));
      name = ['feedback_opt' name];
    else
    name = ['bbci' name];
    end
  end

  setup = control_gui_queue(fig,'get_setup');
  switch typ
   case 'control_player1'
    setu = setup.control_player1;
   case 'control_player2'
    setu = setup.control_player2;
   case 'graphic_player1'
    setu = setup.graphic_player1;
   case 'graphic_player2'
    setu = setup.graphic_player2;
   case 'graphic_master'
    setu = setup.graphic_master;
  end
  eval(sprintf('setu.%s = value;',name));
  if visible
    ind = strmatch(str,setu.fields);
    if isempty(ind)
      ind = length(setu.fields);
    end
    ind = ind+1;
    setu.fields = cat(2,setu.fields(1:ind-1),{name},setu.fields(ind:end));
    setu.fields_help = cat(2,setu.fields_help(1:ind-1),{tool},setu.fields_help(ind:end));
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
   case 'graphic_master'
    setup.graphic_master = setu; 
  end
  control_gui_queue(fig,'set_setup',setup);
  
  if visible
    switch typ
      case 'control_player1'
       plot_control_gui(fig,1);
      case 'control_player2'
       plot_control_gui(fig,2);
      case 'graphic_player1'
       plot_graphic_gui(fig,1);
      case 'graphic_player2'
       plot_graphic_gui(fig,2);
      case 'graphic_master'
       plot_master_gui(fig);
    end
  end
  
end


% enable gui
set(fig,'Visible','on');
