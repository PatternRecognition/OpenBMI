set_general_port_fields('localhost');
VP_CODE = 'ccc'
%bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
%pause


system('cmd /C "c: & cd\ & cd bbci\pyff\src & python FeedbackController.py " &')
%bvr_sendcommand('viewsignals');
pause(10)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);


% p300 oddball. tactile
waitfor(msgbox('Count the tactile stimulus'));
send_xmlcmd_udp('interaction-signal', 's:_feedback','P300Oddball','command','sendinit');
pause(3);
send_xmlcmd_udp('interaction-signal', 'i:screenWidth', 1280, 'i:screenHeight', 800);
send_xmlcmd_udp('interaction-signal', 'i:fullscreen',1);
send_xmlcmd_udp('interaction-signal', 'i:tactile_on',1);
send_xmlcmd_udp('interaction-signal', 'i:visual_on',0);
send_xmlcmd_udp('interaction-signal', 'i:output_device', 4)
send_xmlcmd_udp('interaction-signal', 'i:num_trials',200);



send_xmlcmd_udp('interaction-signal', 'command', 'play');
pause
waitfor(msgbox('Count the visual stimulus'));



% p300 oddball. visual
send_xmlcmd_udp('interaction-signal', 'i:visual_on',1);
send_xmlcmd_udp('interaction-signal', 'i:tactile_on',0);

send_xmlcmd_udp('interaction-signal', 'command', 'play');

pause

% tactile stimulus. keypress
send_xmlcmd_udp('interaction-signal', 's:_feedback','P300Tactile','command','sendinit');
pause(3);
waitfor(msgbox('Press the space bar each time you feel the target stimulus'));
send_xmlcmd_udp('interaction-signal', 'i:screenWidth', 1280, 'i:screenHeight', 800);
send_xmlcmd_udp('interaction-signal', 'i:fullscreen',1);
send_xmlcmd_udp('interaction-signal', 's:datafile_prefix',[VP_CODE '_tact_keypress_']);
send_xmlcmd_udp('interaction-signal', 'i:ask_count',0)
send_xmlcmd_udp('interaction-signal', 'i:tactile_stimulus',1);
send_xmlcmd_udp('interaction-signal', 'i:visual_stimulus',0);
send_xmlcmd_udp('interaction-signal', 'i:output_device', 4)
send_xmlcmd_udp('interaction-signal', 'i:nr_sequences',1);
send_xmlcmd_udp('interaction-signal', 'i:max_trials', 1)

send_xmlcmd_udp('interaction-signal', 'command', 'play');

pause
waitfor(msgbox('Press the space bar each time you see the target stimulus'));
% visual stimulus. keypress
send_xmlcmd_udp('interaction-signal', 's:datafile_prefix',[VP_CODE '_vis_keypress_']);
send_xmlcmd_udp('interaction-signal', 'i:tactile_stimulus',0);
send_xmlcmd_udp('interaction-signal', 'i:visual_stimulus',1);
send_xmlcmd_udp('interaction-signal', 'command', 'play');

pause

% tactile stimulus. count
waitfor(msgbox('Count the tactile stimulus. You will need to enter the number at the end'));
send_xmlcmd_udp('interaction-signal', 'i:ask_count',1)
send_xmlcmd_udp('interaction-signal', 's:datafile_prefix',[VP_CODE '_tact_count_']);
send_xmlcmd_udp('interaction-signal', 'i:nr_sequences',1);
send_xmlcmd_udp('interaction-signal', 'i:max_trials', 1)
send_xmlcmd_udp('interaction-signal', 'i:tactile_stimulus',1);
send_xmlcmd_udp('interaction-signal', 'i:visual_stimulus',0);
send_xmlcmd_udp('interaction-signal', 'command', 'play');

pause

% visual stimulus. count
waitfor(msgbox('Count the visual stimulus. You will need to enter the number at the end'));
send_xmlcmd_udp('interaction-signal', 's:datafile_prefix',[VP_CODE '_tact_count_']);
send_xmlcmd_udp('interaction-signal', 'i:tactile_stimulus',0);
send_xmlcmd_udp('interaction-signal', 'i:visual_stimulus',1);
send_xmlcmd_udp('interaction-signal', 'command', 'play');

fprintf('Finished');
pause


