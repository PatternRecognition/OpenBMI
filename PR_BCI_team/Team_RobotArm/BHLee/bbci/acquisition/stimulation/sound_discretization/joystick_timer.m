function timerCallback(timerObj,event,str_arg)
% this function is executed every time the timer object triggers

% read the coordinates

coords = get(0,'PointerLocation');
do=get(timerObj,'UserData');
do_new=[do;coords];
set(timerObj,'UserData',do_new);
% print the coordinates to screen
fprintf('x: %4i y: %4i\n',coords)

end % function
