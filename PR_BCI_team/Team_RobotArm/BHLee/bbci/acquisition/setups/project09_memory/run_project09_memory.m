global IO_ADDR
fbname= 'VocabularyDeveloperFeedback';
part=5;

fb_opt=[];
fb_opt.dir='D:\svn\pyff\src';
fb_opt.a='D:\svn\bbci\python\pyff\src\Feedbacks';
fb_opt.port=num2str(dec2hex(IO_ADDR));
fb_opt.gui=1;

pyff('startup',fb_opt)
bvr_sendcommand('viewsignals');
% pause(4)
fprintf('Going to real recording now.\n');

if part==1 || part==2 || part == 4 || part == 5
bvr_startrecording([sprintf('memory%s_train%i',VP_CODE,part)])
else
bvr_startrecording([sprintf('memory%s_test%i',VP_CODE,part)])
end
% pause(5)
pyff('init',fbname)

pause(1)
% modify vpcode to include date so as to store all data in a common
% Today_dir, else logfiles are stored in vp_code dir
% today_vec= clock;
% today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
% logfileDir = [VP_CODE '_' today_str];

pyff('set','store_path', TODAY_DIR);
pyff('set','VP',VP_CODE)
pyff('setint','part', part);
pause(0.5);
pyff('play');

stimutil_waitForInput('phrase', 'go', ...
       'msg', 'When run has finished, give fokus to Matlab terminal and input "go<RET>".');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');