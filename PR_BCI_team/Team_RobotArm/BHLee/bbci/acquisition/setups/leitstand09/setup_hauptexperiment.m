%% Settings for Leitstand (AlarmControl)
common_stop_time = 45*60;

%% Graphical settings
main_gui_size = [800,600];
main_gui_pos = [0,100];
fail_gui_size = [400,300];
fail_gui_pos = [800,100];

%% Send to pyff
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'AlarmControl','command','sendinit');  % Choose Feedback
% send_xmlcmd_udp('interaction-signal', 'time_scale',time_scale);
% send_xmlcmd_udp('interaction-signal', 'i:tick_frequency',tick_frequency);
send_xmlcmd_udp('interaction-signal', 'i:common_stop_time',common_stop_time);
% send_xmlcmd_udp('interaction-signal', 'i:money_report_interval',money_report_interval);
% send_xmlcmd_udp('interaction-signal', 'min_time',min_time);
% send_xmlcmd_udp('interaction-signal', 'max_time',max_time);
% send_xmlcmd_udp('interaction-signal', 'min_money_per_trial',min_money_per_trial);
% send_xmlcmd_udp('interaction-signal', 'max_money_per_trial',max_money_per_trial);

send_xmlcmd_udp('interaction-signal', 'i:main_gui_size',[VP_SCREEN(3)-20,VP_SCREEN(4)-20]);
send_xmlcmd_udp('interaction-signal', 'i:common_main_gui_pos',[VP_SCREEN(1)+600,200]);
send_xmlcmd_udp('interaction-signal', 'i:fail_gui_size',[800,500]);
send_xmlcmd_udp('interaction-signal', 'i:common_fail_gui_pos',[2600,200]);
