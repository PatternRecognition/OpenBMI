%%- BCI01 - Test run
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --loglevel=debug --plugin=FeedbackControllerPlugins --additional-feedback-path=d:\svn\pyff_external_feedbacks" &')
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);


fprintf('TESTRUN BCI 01\nPress <RET> to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300hex');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to continue with real measurement.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');

fprintf('Kill Windows Command Window and press <RET>\n');
pause

%%- BCI01 - Run with Recording
isi= 450;

system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --loglevel=debug --plugin=FeedbackControllerPlugins --additional-feedback-path=d:\svn\pyff_external_feedbacks" &')
pause(3)

fprintf('Real Recording BCI 01\nPress <RET> to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',['visualhex_p300_' int2str(isi) 'ms']);
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'p300hex');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
send_xmlcmd_udp('interaction-signal', 'time_in_between', isi);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');



%%- BCI02 - Test run
setup_biomed08_BCI02
fprintf('\nTESTRUN BCI 02\nPress <RET> to start.\n');
pause
stim_tactileP300(6*5, 5, opt, 'test',1, 'visual_targetPresentation', 1);

%%- BCI02 - Run with recording
setup_biomed08_BCI02
fprintf('Real Recording BCI 02\nPress <RET> to start.\n');
pause
stim_tactileP300(6*10, 20, opt)
%stim_tactileP300(6*10, 20, opt, 'visual_targetPresentation', 1, 'filename', 'tactilehex_p300_visuresp');
%stim_tactileP300(6*10, 20, opt, 'filename', 'tactilehex_p300_resp');
%opt.speakerSelected= [1:5];

%%- BCI 03 - Test run
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --loglevel=debug --plugin=FeedbackControllerPlugins --additional-feedback-path=d:\svn\pyff_external_feedbacks" &')
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);


fprintf('TESTRUN BCI03\nPress <RET> to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AudioHexOSpell');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to continue with real measurement.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('Kill Windows Command Window and press <RET>\n');
pause


%%- BCI 03 - Run with real Recording
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --loglevel=debug --plugin=FeedbackControllerPlugins --additional-feedback-path=d:\svn\pyff_external_feedbacks" &')
pause(5)

isi= 300;
fprintf('Real Recording BCI 03\nPress <RET> to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',['auditoryhex_p300_' int2str(isi) 'ms']);
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AudioHexOSpell');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit'); pause(0.5);
%send_xmlcmd_udp('interaction-signal', 'letter_timeout', isi);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


%%- BCI 04 - Test run
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --loglevel=debug --plugin=FeedbackControllerPlugins --additional-feedback-path=d:\svn\pyff_external_feedbacks" &')
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);


fprintf('TESTRUN BCI04\nPress <RET> to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to continue with real measurement.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


fprintf('Kill Windows Command Window and press <RET>\n');
pause


%%- BCI 04 - Run with real Recording
system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py --loglevel=debug --plugin=FeedbackControllerPlugins --additional-feedback-path=d:\svn\pyff_external_feedbacks" &')
pause(5)

isi= 275;
fprintf('Real Recording BCI 04\nPress <RET> to start.\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',['visual_p300_speller_' int2str(isi) 'ms']);
%send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',['visual_p300_speller']);
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit'); pause(0.5);
%send_xmlcmd_udp('interaction-signal', 'letter_timeout', isi);
pause
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> when finished.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
