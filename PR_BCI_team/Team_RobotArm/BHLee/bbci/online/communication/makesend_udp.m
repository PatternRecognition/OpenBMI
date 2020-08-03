if isunix
    mex send_udp.c udp.c 
    for i = 1:5
      system(sprintf('cp send_udp.mexglx send_udp%d.mexglx',i));
    end
    
else
    mex send_udp.c udp.c ws2_32.lib 
    for i = 1:5
      system(sprintf('copy send_udp.dll send_udp%d.dll',i));
    end
end