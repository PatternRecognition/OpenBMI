function change_bias(bias)
global general_port_fields
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
send_xmlcmd_udp('interaction-signal', ...
    'threshold', bias);
end