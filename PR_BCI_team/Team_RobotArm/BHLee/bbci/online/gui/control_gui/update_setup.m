function update_setup(fig,player);
% UPDATE_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% updates a setup in the gui
%
% usage:
%    update_setup(fig,player);
% 
% input:
%    fig     the handle of the gui
%    player  player number
%
% Guido Dornhege
% $Id: update_setup.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

set(fig,'Visible','off');drawnow;
setup = control_gui_queue(fig,'get_setup');

%open a listening port 
port = 12400+player;

update_comm = get_data_udp(port);

if player==1
  send_evalstring(setup.control_player1.machine,setup.control_player1.port,sprintf('get_actual_setup_list = {''%s'',%d};',ma,port));
else
  send_evalstring(setup.control_player2.machine,setup.control_player2.port,sprintf('get_actual_setup_list = {''%s'',%d};',ma,port));
end

% get the answer 
tic;
a = [];

while isempty(a) & toc<2
  a = get_data_udp(update_comm,0);
  pause(0.05);
end

% close the port 
get_data_udp(update_comm);

if isempty(a)
  set(fig,'Visible','on');
  return;
end

a = char(a);
str = {};

if ~strcmp(a,'.')

c = strfind(a,',');
while ~isempty(c)
  str = {str{:},a(1:c(1)-1)};
  a = a(c(1)+1:end);
  c = strfind(a,',');
end
if ~isempty(a)
  str = {str{:},a};
end
end

eval(sprintf('setup.general.setup_list%d = str;',player));
eval(sprintf('setup.general.setup_list_default%d = str;',player));

control_gui_queue(fig,'set_setup',setup);

% add by Claudia. Copied from im_lucky
[dum1,dum2,dum3,classes] = get_directory_info([],dire,player,machine,port)
cltmp = char(classes);
idx = findstr(cltmp,'/');
setup.graphic_player1.feedback_opt.classes{1} = cltmp(1:idx-1);
setup.graphic_player1.feedback_opt.classes{2} = cltmp(idx+1:end);
control_gui_queue(fig,'set_setup',setup);

ax = control_gui_queue(fig,'get_general_ax');

eval(sprintf('h = ax.setup_list%d;',player));

set(h,'String',str);
set(h,'Value',1);

set(fig,'Visible','on');
  





