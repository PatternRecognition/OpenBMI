function stop_control_setup(fig,typ)
% STOP_CONTROL_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% send to all applications the stop signal
%
% usage:
%    stop_control_setup(fig,typ);
% 
% input:
%    fig   the handle of the gui
%    typ   control_player1, control_player2, graphic_player1, graphic_player2 
%          or all 
%          submit stop to the specific or to all applications
%
% Guido Dornhege
% $Id: stop_control_setup.m,v 1.2 2007/03/26 12:23:28 neuro_cvs Exp $

% actual entries
activate_all_entries(fig);
setup = control_gui_queue(fig,'get_setup');

if setup.savemode==true
  interrupt_saving(fig);
  setup = control_gui_queue(fig,'get_setup');
end

% submit stop depending of the typ
switch typ
 case 'all'
  send_informations(fig,'control_player1','run=false;loop=false;');
  if setup.general.player>1
    send_informations(fig,'control_player2','run=false;loop=false;');
  end
  
  if setup.general.graphic
    send_informations(fig,'graphic_player1','run=false;loop=false;');
    if setup.general.player>1
      send_informations(fig,'graphic_player2','run=false;loop=false;');
    end
    
  end
  send_informations(fig,'graphic_master','run=false;loop=false;');

 case 'control_player1'
  send_informations(fig,'control_player1','run=false;loop=false;');

 case 'control_player2'
  send_informations(fig,'control_player2','run=false;loop=false;');

 case 'graphic_player1'
  send_informations(fig,'graphic_player1','run=false;loop=false;');

 case 'graphic_player2'
  send_informations(fig,'graphic_player2','run=false;loop=false;');  
  
 case 'graphic_master'
  send_informations(fig,'graphic_master','run=false;loop=false;');  
end
