%% Settings for pyff oddball (P300_rectangle)

send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300_Rectangle','command','sendinit');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
send_xmlcmd_udp('interaction-signal', 'i:screen_pos',VP_SCREEN);
send_xmlcmd_udp('interaction-signal', 'i:stim_duration',1500);

