%% Settings for pyff nback (nback_verbal)

stimTime = 12;
preResponseTime = 0;
responseTime = 180;

test_symbols = '*';
test_nOccur = 3;        % nr occurences of each letter
test_nMatch = 1;

send_xmlcmd_udp('interaction-signal', 's:_feedback', 'nback_verbal','command','sendinit');  % Choose Feedback
send_xmlcmd_udp('interaction-signal', 'i:screenPos',VP_SCREEN);
send_xmlcmd_udp('interaction-signal', 'i:size',120);
send_xmlcmd_udp('interaction-signal', 'i:auditoryFeedback',0);
send_xmlcmd_udp('interaction-signal', 'i:nCountdown',3);

send_xmlcmd_udp('interaction-signal', 'i:triggers',10:9+numel(test_symbols));
send_xmlcmd_udp('interaction-signal', 's:symbols',test_symbols);
send_xmlcmd_udp('interaction-signal', 'i:nOccur',test_nOccur);
send_xmlcmd_udp('interaction-signal', 'i:nMatch',test_nMatch);
send_xmlcmd_udp('interaction-signal', 'i:n',n);

send_xmlcmd_udp('interaction-signal', 'i:stimTime',stimTime);
send_xmlcmd_udp('interaction-signal', 'i:preResponseTime',preResponseTime);
send_xmlcmd_udp('interaction-signal', 'i:responseTime',responseTime);