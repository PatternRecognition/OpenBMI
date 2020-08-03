if isunix
    mex send_to_udp.c udp.c 
else
    mex send_to_udp.c udp.c ws2_32.lib 
end