function move_up(fig,typ,str);
% MOVE_UP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% moves the element in the gui one step up
%
% usage:
%    move_up(fig,typ,str);
% 
% input:
%    fig     the handle of the gui
%    typ     control_player1,control_player2,graphic_player1,graphic_player2
%    str     the fieldname to move up
%
% Guido Dornhege
% $Id: move_up.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% actual setup and position
setup = control_gui_queue(fig,'get_setup');

eval(sprintf('setu = setup.%s;',typ));

ind = find(strcmp(str,setu.fields));

if ind==1
  perm = [2:length(setu.fields),1];
else
  perm = 1:length(setu.fields);
  perm(ind) = ind-1;
  perm(ind-1) = ind;
end

% permute it
setu.fields = setu.fields(perm);
setu.fields_help = setu.fields_help(perm);

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

% change the figure
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
