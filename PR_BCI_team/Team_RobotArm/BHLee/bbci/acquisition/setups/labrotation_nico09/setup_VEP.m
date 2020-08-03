%% Settings for pyff VEP (CheckerboardVEP)

% bei 150 stim und 1.5s/stimulus ~ 5 min
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CheckerboardVEP','command','sendinit');
send_xmlcmd_udp('interaction-signal', 'i:stim_duration', 1500);
send_xmlcmd_udp('interaction-signal', 'i:squaresPerSide',10);
send_xmlcmd_udp('interaction-signal', 'i:countdown_from',3);
send_xmlcmd_udp('interaction-signal', 'i:screen_pos',VP_SCREEN);
send_xmlcmd_udp('interaction-signal', 'i:nStim_per_block',50);
send_xmlcmd_udp('interaction-signal', 'i:nStim',150);
