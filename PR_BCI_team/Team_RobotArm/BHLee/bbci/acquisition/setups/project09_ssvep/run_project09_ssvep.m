% part 1
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --port=0x2030 --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')

bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
fprintf('Going to real recording now.\n');
fbname= 'training_ssvep_july';
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',fbname);
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'training_ssvep_july');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(0.5);
send_xmlcmd_udp('interaction-signal', 's:datafilename', [TODAY_DIR 'responses.txt']);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
stimutil_waitForInput('phrase', 'go', ...
       'msg', 'When run has finished, give fokus to Matlab terminal and input "go<RET>".');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

% % part 1b
% system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug --port=0x2030 --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')
% bvr_sendcommand('viewsignals');
% pause(5)
% send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
% fprintf('Going to real recording now.\n');
% fbname= 'Hex_O';
% send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',fbname);
% send_xmlcmd_udp('interaction-signal', 's:_feedback', 'Hex_O');
% send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
% pause(0.5);
% send_xmlcmd_udp('interaction-signal', 's:datafilename', [TODAY_DIR 'responses.txt']);
% send_xmlcmd_udp('interaction-signal', 'command', 'play');
% stimutil_waitForInput('phrase', 'go', ...
%        'msg', 'When run has finished, give fokus to Matlab terminal and input "go<RET>".');
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');

% % part 2
% system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins  --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')
% bvr_sendcommand('viewsignals');
% pause(5)
% send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
% % fprintf('Going to real recording now.\n');
% % fbname= 'reaction1';
% % send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',fbname);
% % send_xmlcmd_udp('interaction-signal', 's:_feedback', 'reaction1');
% % send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
% pause(0.5);
% send_xmlcmd_udp('interaction-signal', 's:datafilename', [TODAY_DIR 'responses.txt']);
% send_xmlcmd_udp('interaction-signal', 'command', 'play');
% stimutil_waitForInput('phrase', 'go', ...
%        'msg', 'When run has finished, give fokus to Matlab terminal and input "go<RET>".');
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');












