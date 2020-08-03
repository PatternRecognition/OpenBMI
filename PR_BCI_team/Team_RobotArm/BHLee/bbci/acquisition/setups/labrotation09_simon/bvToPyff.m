

response_marker = 'R 16';
disp(['Using response marker "' response_marker '"']);
state= acquire_bv(100, general_port_fields.bvmachine);
[dmy]= acquire_bv(state);  %% clear the queue
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
while 1 
    try
    [dmy,bn,mp,mt,md]= acquire_bv(state);
    catch
      % this prevents the program from terminating when the bvr switches 
      % from monitoring the eeg to saving data
      pause(2)
      state= acquire_bv(100, general_port_fields.bvmachine);
    end
    if sum(strcmp(mt, response_marker))>0
      send_xmlcmd_udp('interaction-signal', ...
        's:keypress', 'yes');
    end
    pause(0.01);  %% this is to allow breaks
end
