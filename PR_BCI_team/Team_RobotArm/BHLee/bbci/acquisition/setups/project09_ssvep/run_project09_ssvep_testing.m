% part 1

general_port_fields.bvmachine='127.0.0.1';
general_port_fields.control{1}='127.0.0.1';
[dum,host]=unix('hostname');
host=host(1:end-1);
if strcmp(host,'tubbci2')
  fprintf('port is set to: 0x%s\n',dec2hex(IO_ADDR))
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --nogui --port=0x2030 -l debug --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')
%system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --port=0x2030 -l debug --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')

else
  warning('port could be wrong, check if markers arrive')
  system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --nogui --port=0x5C00 -l debug --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')
%system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --port=0x5C00 -l debug --additional-feedback-path=D:\svn\bbci\python\pyff\src\Feedbacks" &')

end

%bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
fprintf('Going to real recording now.\n');

fbname= 'Hex_O';
send_xmlcmd_udp('interaction-signal', 's:_feedback', fbname,'command','sendinit');
send_xmlcmd_udp('interaction-signal', 's:store_path', TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 's:VP', VP_CODE);

pause(3);
%bvr_sendcommand('startrecording', 's:BASENAME','training')
send_xmlcmd_udp('interaction-signal', 'command', 'play');
stimutil_waitForInput('phrase', 'go','msg', 'When run has finished, give fokus to Matlab terminal and input "go<RET>".');
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
