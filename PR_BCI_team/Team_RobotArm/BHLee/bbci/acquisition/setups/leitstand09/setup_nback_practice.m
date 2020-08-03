%% Settings for pyff nback (nback_verbal)

stimTime = 12;
preResponseTime = 0;
responseTime = 180;

practice_symbols = 'ABC';
practice_nOccur = 4;        % nr occurences of each letter
practice_nMatch = 1;

send_xmlcmd_udp('interaction-signal', 's:_feedback', 'nback_verbal','command','sendinit');
send_xmlcmd_udp('interaction-signal', 'i:screenPos',VP_SCREEN);
send_xmlcmd_udp('interaction-signal', 'i:size',120);
send_xmlcmd_udp('interaction-signal', 'i:auditoryFeedback',1);
send_xmlcmd_udp('interaction-signal', 'i:nCountdown',3);


send_xmlcmd_udp('interaction-signal', 'i:triggers',10:9+numel(practice_symbols));
send_xmlcmd_udp('interaction-signal', 's:symbols',practice_symbols);
send_xmlcmd_udp('interaction-signal', 'i:nOccur',practice_nOccur);
send_xmlcmd_udp('interaction-signal', 'i:nMatch',practice_nMatch);
send_xmlcmd_udp('interaction-signal', 'i:n',n);

send_xmlcmd_udp('interaction-signal', 'i:stimTime',stimTime);
send_xmlcmd_udp('interaction-signal', 'i:preResponseTime',preResponseTime);
send_xmlcmd_udp('interaction-signal', 'i:responseTime',responseTime);