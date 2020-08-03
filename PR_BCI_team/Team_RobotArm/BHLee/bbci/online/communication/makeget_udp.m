if isunix
    mex get_udp.c udp.c 
    for i = 1:5
      system(sprintf('cp get_udp.mexglx get_udp%d.mexglx',i));
    end
else
    mex get_udp.c udp.c ws2_32.lib
    for i = 1:5
      system(sprintf('copy get_udp.dll get_udp%d.dll',i));
    end
end 