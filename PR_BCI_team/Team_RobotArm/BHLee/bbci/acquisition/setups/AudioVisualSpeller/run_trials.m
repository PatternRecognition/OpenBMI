function run_trials(desired_target, desired_cond)
%RUN_TRIAL starts a trial in specified condition

if max(desired_cond) < 5
    string_desired_target = strrep(num2str(desired_target), ' ', '');
    string_desired_cond = strrep(num2str(desired_cond), ' ', '');

    send_xmlcmd_udp('interaction-signal', 's:desired_target' , string_desired_target)
    send_xmlcmd_udp('interaction-signal', 's:desired_cond' , string_desired_cond)
else
my_letters = ['A':'Z' '0':'9'];
    letter_target = my_letters(desired_target);
    string_desired_cond = strrep(num2str(desired_cond), ' ', '');
    
    send_xmlcmd_udp('interaction-signal', 's:desired_target' , letter_target)
    send_xmlcmd_udp('interaction-signal', 's:desired_cond' , string_desired_cond)
end
pause(.5)
send_xmlcmd_udp('interaction-signal', 'command', 'play');
%wait untill trial ends
stimutil_waitForMarker(255)
sprintf('stopping the recording!')
bvr_sendcommand('stoprecording')
