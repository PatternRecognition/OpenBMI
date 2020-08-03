function m_timer(timerObj,event,log_fid)
% this function is executed every time the timer object triggers

j_val=jst;
log_fid=get(timerObj,'UserData');
fprintf(log_fid,'Time= %s Joystick_value= %f Marker= %i \n',datestr(now, 'HH:MM:SS.FFF'),j_val(2),0);  
% read the coordinates
end % function
