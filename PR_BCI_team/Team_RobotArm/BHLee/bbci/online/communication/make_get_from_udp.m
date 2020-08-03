if isunix
    mex get_from_udp.c udp.c 
else
    mex get_from_udp.c udp.c ws2_32.lib
end 