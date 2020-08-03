function record_joystick_movements(ti,opt)
global VP_CODE
opt= set_defaults(opt, ...
                  'joystick',0, ...
                  'break',8, ...
                  'test', 0, ...
                  'require_response', 1, ...
                  'background', 0.5*[1 1 1], ...
                  'countdown', 5, ...
                  'countdown_fontsize', 0.3, ...
                  'duration_intro', 4000, ...
                  'bv_host', 'localhost', ...
                  'msg_intro','Entspannen', ...
                  'msg_fin', 'Ende', ...
                  'mssg',[], ...
                  'filename','joystick_movements')
              
[h_msg, opt.handle_background]= stimutil_initMsg;
desc= stimutil_readDescription('joystick_movements');
h_desc= stimutil_showDescription(desc, 'waitfor',10);
%opt.delete_obj= h_desc.axis;

drawnow;
waitForSync;


 if ~isempty(opt.bv_host),
  bvr_checkparport;
end


% set(h_msg, 'String',opt.msg_intro, 'Visible','on');
% drawnow;
% waitForSync;

if opt.test,
  fprintf('Warning: test option set true: EEG is not recorded!\n');
else
  if ~isempty(opt.filename),
    bvr_startrecording([opt.filename VP_CODE]);
  else
    warning('!*NOT* recording: opt.filename is empty');
  end
  ppTrigger(251);
  waitForSync(8);
end

%open log file for recording joystick data
log_fid=open_joystick_log('joystick_movements');
stimutil_countdown(opt.countdown);   
%initialize timer
timerObj = timer('TimerFcn',@m_timer,'Period',0.02,'ExecutionMode','fixedRate','UserData',log_fid);
j_val=jst;
fprintf(log_fid,'Time= %s Joystick_value= %f Marker= %i \n',datestr(now, 'HH:MM:SS.FFF'),j_val(2),1);
start(timerObj); 
pause(ti)
stop(timerObj);
fprintf(log_fid,'Time= %s Joystick_value= %f Marker= %i \n',datestr(now, 'HH:MM:SS.FFF'),j_val(2),2);
fclose(log_fid)

%set(opt.handle_cross, 'Visible','off'); 
set(h_msg, 'Visible','on');
set(h_msg, 'String',opt.msg_fin);


ppTrigger(254);
pause(1);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end

pause(2);
delete(h_msg);
