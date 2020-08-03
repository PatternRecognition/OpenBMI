function activate_this_entry(fig,field,value);
% ACTIVATE_THIS_ENTRY ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% activate the specific entry of the gui
%
% usage:
%      activate_this_entry(fig,field,value);
%
% input:
%     fig   the handle of the gui
%     field the fieldname
%     value the value of the field
%
% Guido Dornhege
% $Id: activate_this_entry.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% get actual setup
setup = control_gui_queue(fig,'get_setup');

% try to change and save or error handling
try
  eval(sprintf('%s = eval(value);',field));
catch
  set(fig,'Visible','off');
  drawnow;
  message_box('Check the entries',1);
  set(fig,'Visible','on');
  return;
end

control_gui_queue(fig,'set_setup',setup);
