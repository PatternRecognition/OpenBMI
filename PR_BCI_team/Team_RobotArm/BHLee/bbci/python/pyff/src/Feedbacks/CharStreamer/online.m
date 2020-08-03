% matlab script for Online CharStreamer feedback
pause on;
% TODO brainrecorder stuff

send_xmlcmd_udp('init', '127.0.0.1', 12345)
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'CharStreamer','command','sendinit')


%% IMPORTANT !wait till feedback is completely loaded
pause(1)
%stimutil_waitForMarker(93)

%% configuration
send_xmlcmd_udp('interaction-signal', 'b:calibration_mode' , 'False');
send_xmlcmd_udp('interaction-signal', 'b:online_mode' , 'True');
send_xmlcmd_udp('interaction-signal', 'i:iterations' , 15);
send_xmlcmd_udp('interaction-signal', 'i:pre_iterations' , 0); % fake iterations before

send_xmlcmd_udp('interaction-signal', 'b:early_stopping' , 'True');
send_xmlcmd_udp('interaction-signal', 'i:min_iterations', 6); % for early stopping
send_xmlcmd_udp('interaction-signal', 'f:p_criterion', 0.05);


%% simulating
send_xmlcmd_udp('interaction-signal', 'b:online_simulation' , 'True');

% start - !!! feedback only stops in simulation mode, otherwise it loops
% until stop is hit
for t = ['h', 'a', 'l', 'l', 'o']
    send_xmlcmd_udp('interaction-signal', 's:target', t); % works only for simulation and calibration
    pause(1)
    send_xmlcmd_udp('interaction-signal', 'command', 'play'); % so this isn't needed in normal online mode
    pause
end

%% stop
send_xmlcmd_udp('interaction-signal', 'command', 'stop');

% quit
send_xmlcmd_udp('interaction-signal', 'command', 'quit');