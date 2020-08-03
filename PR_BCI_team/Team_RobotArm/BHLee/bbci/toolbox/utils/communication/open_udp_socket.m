function sock = main
% Open and return an udp socket

    sock = pnet('udpsocket', 1206);
    pnet(sock, 'setreadtimeout', 0);

end
