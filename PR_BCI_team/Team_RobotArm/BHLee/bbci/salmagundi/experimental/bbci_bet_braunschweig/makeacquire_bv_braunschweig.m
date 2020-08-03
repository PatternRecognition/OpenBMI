if isunix
    mex acquire_bv_braunschweig.c brainserver_braunschweig.c headerfifo_braunschweig.c ../bbci_bet_unstable/winunix/winthreads.c ../bbci_bet_unstable/winunix/winevents.c -lrt
else
    mex acquire_bv_braunschweig.c brainserver_braunschweig.c headerfifo_braunschweig.c ws2_32.lib
end