VP_CODE= 'test';
acq_makeDataFolder
VP_SCREEN = [-1275         229        1272         741]
system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ... 
  ' -a D:\svn\bbci\python\pyff\src\Feedbacks  --nogui -l debug -p brainvisionrecorderplugin" &']);
pause(7);

send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CheckerboardVEP','command','sendinit');
send_xmlcmd_udp('interaction-signal', 'i:stim_duration', 1500);
send_xmlcmd_udp('interaction-signal', 'i:squaresPerSide',10);
send_xmlcmd_udp('interaction-signal', 'i:countdown_from',3);
send_xmlcmd_udp('interaction-signal', 'i:screen_pos',VP_SCREEN);
send_xmlcmd_udp('interaction-signal', 'i:nStim_per_block',50);
send_xmlcmd_udp('interaction-signal', 'i:nStim',150);

send_xmlcmd_udp('interaction-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','VEP_');
pause(5)

send_xmlcmd_udp('interaction-signal', 'command', 'play');

file= [TODAY_DIR 'VEP_' VP_CODE];
mrk= eegfile_readBVmarkers(file);

file= [TODAY_DIR 'RSVP_Color83mstest02']
mrk= eegfile_readBVmarkers(file);
