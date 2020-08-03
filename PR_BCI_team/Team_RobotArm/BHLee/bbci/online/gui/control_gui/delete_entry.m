function delete_entry(fig,typ,str);
% DELETE_ENTRY ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% deletes an entry of a gui
%
% usage:
%    delete_entry(fig,typ,str);
%
% input:
%    fig   the handle of the gui
%    typ   control_player1, control_player2, graphic_player1, graphic_player2
%    str   the fieldname to delete
% 
% Guido Dornhege
% $Id: delete_entry.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% get actual setup
setup = control_gui_queue(fig,'get_setup');

eval(sprintf('setu = setup.%s;',typ));

% find field position and delete
ind = find(strcmp(str,setu.fields));

setu.fields(ind) = [];
setu.fields_help(ind) = [];

setu = remove_field(setu,str);


% save changes
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

% plot changes
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
