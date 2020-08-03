function exit_control_setup(fig);
% EXIT_CONTROL_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% exits the gui and stop all applications
%
% usage:
%    exit_control_setup(fig);
%
% input:
%    fig    the handle of the gui
%
% Guido Dornhege
% $Id: exit_control_setup.m,v 1.2 2006/11/23 15:42:15 neuro_cvs Exp $

% stop timer
update_timer;

% stop saving
interrupt_saving(fig);

% send stop signal
stop_control_setup(fig,'all');

% close the figure
close_control_figure(fig);

