bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -p FeedbackControllerPlugins" &')
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins  --additional-feedback-path=D:\svn\pyff_external_feedbacks" &')
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

%testrun
fprintf('Press <RET> to start the test-runs.\n');
pause()

fprintf('TESTRUN\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AuditoryOddball');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to continue with tests measurement.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fprintf('TESTRUN2\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_OrientationPerformance');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Still OK?? Volume etc? .\n');
pause()
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fprintf('Press <RET> to go for the real measurement.\n');
pause()

fprintf('AO_ArtefactsSelfpace\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_selfpaced');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_ArtefactsSelfpace');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fprintf('AO_ToneHeightPerformance\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_toneheight');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_ToneHeightPerformance'); pause(0.5);
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('AO_OrientationPerformance\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_orientation');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_OrientationPerformance');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('AO_Auditory\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_auditory');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_Auditory');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fprintf('AO_Auditory\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_auditory');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_Auditory');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('AO_Visual\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_visual');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_Visual');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('AO_Auditory\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_auditory');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_Auditory');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the next run.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('AO_Speller\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME','ao_speller');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AO_Speller');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause(1)
send_xmlcmd_udp('interaction-signal', 's:save_folder',TODAY_DIR);
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to finish.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fprintf('Finshed.\n');


