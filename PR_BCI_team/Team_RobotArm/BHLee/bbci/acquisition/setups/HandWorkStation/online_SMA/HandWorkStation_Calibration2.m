%% Initialize communication by UDP
% IP of SmA PC: 192.168.1.23
send_xmlcmd_udp('init', '192.168.1.23', 12345);

nBlocks = length(ISI_list);
speed = 0;
fprintf('starting HandWorkStation - CALIBRATION 2.\n');
for k = 1:nBlocks,
  speed = 1 + mod(speed, 2);
  ppTrigger(speed*10);
  send_xmlcmd_udp('i:bbci_act_output', speed_level(speed));
  pause(ISI_list(k));
  ppTrigger(speed*10+1);
  pause(.1)
end
ppTrigger(255);
