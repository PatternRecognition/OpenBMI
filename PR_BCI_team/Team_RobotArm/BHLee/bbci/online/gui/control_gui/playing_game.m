function playing_game(fig,typ,string)
% PLAYING_GAME ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
% 
% sends play and pause information
%
% usage:
%    playing_game(fig,typ,string)
%
% input:
%    fig    the handle of the gui
%    typ    control_player1, control_player2, graphic_player1, 
%           graphic_player2 or all
%    string play or pause
%
% Guido Dornhege
% $Id: playing_game.m,v 1.2 2007/03/26 12:23:28 neuro_cvs Exp $

% get actual setup
activate_all_entries(fig);
setup = control_gui_queue(fig,'get_setup');


% submit play and pause depending on typ 
switch typ
 case 'all'
  send_informations(fig,'control_player1',['bbci.status=''' string '''']);
  if setup.general.player>1
    send_informations(fig,'control_player2',['bbci.status=''' string '''']);
  end
  
  if setup.general.graphic & ~setup.general.graphic_master
    send_informations(fig,'graphic_player1',['feedback_opt.status=''' string '''']);
    if setup.general.player>1
      send_informations(fig,'graphic_player2',['feedback_opt.status=''' string '''']);
    end
    
  end
  if setup.general.graphic_master
    send_informations(fig,'graphic_master',['feedback_opt.status=''' string '''']);
  end
  
 case 'control_player1'
  send_informations(fig,'control_player1',['bbci.status=''' string '''']);

 case 'control_player2'
  send_informations(fig,'control_player2',['bbci.status=''' string '''']);

 case 'graphic_player1'
  if ~setup.general.graphic_master,
    send_informations(fig,'graphic_player1',['feedback_opt.status=''' string '''']);
  end
 case 'graphic_player2'
  if ~setup.general.graphic_master,
    send_informations(fig,'graphic_player2',['feedback_opt.status=''' string '''']);
  end
 case 'graphic_master'
    send_informations(fig,'graphic_master',['feedback_opt.status=''' string '''']);
  
end
