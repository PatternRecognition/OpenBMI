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
bvr_sendcommand('viewsignals');
pause(2)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
fprintf('Going to real recording now.\n');
%need to put record plaz 
fbname= 'Hex_O_Testing';
%fbname= 'training_ssvep_july';
send_xmlcmd_udp('interaction-signal', 's:_feedback', fbname,'command','sendinit');
pause(7)
send_xmlcmd_udp('interaction-signal', 's:store_path', TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 's:VP', VP_CODE);

%bbci_setup= strcat(TODAY_DIR, '/setup_ssvepVPzk');
bbci_setup= [TODAY_DIR, 'setup_ssvepVPgay.mat'];
%load([TODAY_DIR, 'bias.mat'],'bias');
S= load(bbci_setup);
S.bbci.quit_marker= [253 254];
S.bbci.log=0;
%S.bbci.bias = bias;
save(bbci_setup, '-STRUCT','S');
general_port_fields.feedback_receiver= 'pyff';
bvr_startrecording(['feedback_covert_ssvep_10s_' VP_CODE])
send_xmlcmd_udp('interaction-signal', 'command', 'play');

bbci_bet_apply(bbci_setup, 'bbci.fb_port', 12345);
% send_xmlcmd_udp('interaction-signal', 'command', 'quit');
% today_vec= clock;
% today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
