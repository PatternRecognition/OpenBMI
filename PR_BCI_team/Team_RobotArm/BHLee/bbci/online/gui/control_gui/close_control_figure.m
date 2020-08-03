function close_control_figure(fig);
% CLOSE_CONTROL_FIGURE ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% closes the gui
%
% usage:
%     close_control_figure(fig);
%
% input:
%     fig    the handle of the gui
%
% Guido Dornhege
% $ID$

% remove from queue and delete
update_timer;
control_gui_queue(fig,'remove');
delete(fig);
