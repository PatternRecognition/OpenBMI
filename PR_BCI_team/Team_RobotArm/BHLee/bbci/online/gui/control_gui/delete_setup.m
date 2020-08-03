function delete_setup(fig,player);
% DELETE_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% deletes a setup in the gui
%
% usage:
%    add_setup(fig,player);
% 
% input:
%    fig     the handle of the gui
%    player  player number
%
% Guido Dornhege
% $Id: delete_setup.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


ax = control_gui_queue(fig,'get_general_ax');

eval(sprintf('h = ax.setup_list%d;',player));

str = get(h,'String');
if ~iscell(str),
    str= {str};
end
va = get(h,'Value');

if va<=length(str) & va>0
  str = {str{1:va-1},str{va+1:end}};
  if va>length(str)
    va = length(str);
  end
end

set(h,'String',str,'Value',va);

setup = control_gui_queue(fig,'get_setup');
eval(sprintf('setup.general.setup_list%d = str;',player));
control_gui_queue(fig,'set_setup',setup);

