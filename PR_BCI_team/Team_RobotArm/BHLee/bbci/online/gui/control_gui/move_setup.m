function move_setup(fig,player,dire);
% MOVE_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% moves the setup in the gui one step up or down
%
% usage:
%    move_up(fig,player,dire);
% 
% input:
%    fig     the handle of the gui
%    player  player number
%    dire    +1 up, -1 down
%
% Guido Dornhege
% $Id: move_setup.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% actual setup and position
ax = control_gui_queue(fig,'get_general_ax');

eval(sprintf('h = ax.setup_list%d;',player));

str = get(h,'String');
va = get(h,'Value');

if isempty(va) | va==0 | va>length(str);
  return;
end

if dire==1
  if va==1
    perm = [2:length(str),1];
    va = length(str);
  else
    perm = 1:length(str);
    perm(va) = va-1;
    perm(va-1) = va;
    va = va-1;
  end
else
  if va==length(str);
    perm = [length(str),1:length(str)-1];
    va = 1;
  else
    perm = 1:length(str);
    perm(va) = va+1;
    perm(va+1) = va;
    va = va+1;
  end
end

str = str(perm);

set(h,'String',str,'Value',va);

setup = control_gui_queue(fig,'get_setup');
eval(sprintf('setup.setup_list%d = str;',player));
control_gui_queue(fig,'set_setup',setup);
