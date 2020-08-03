
%TESTRUN

settings_bbci= {'bbci.start_marker', 252, ...
                  'bbci.quit_marker', 253, ...
                  'bbci.adaptation.running',0};

bbci_cfy= [TODAY_DIR '/bbci_classifier.mat'];

% bvr_startrecording('ImpDum'); 
% bvr_sendcommand('stoprecording');

%bvr_startrecording(['T9SpellerCalibration' VP_CODE], 'impedances', 0); 
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'OnlineAuditoryP300Speller','command','sendinit');
pause(10);
send_xmlcmd_udp('interaction-signal', 'i:spellerMode', true );
send_xmlcmd_udp('interaction-signal', 'i:simulate_sbj', false );
pause(1)

send_xmlcmd_udp('interaction-signal', 'command', 'play');
bbci_bet_apply(bbci_cfy, settings_bbci{:});

pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');
