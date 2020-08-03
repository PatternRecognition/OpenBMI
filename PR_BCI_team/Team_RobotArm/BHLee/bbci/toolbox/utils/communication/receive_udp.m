function data = foo(sock)

    len = pnet(sock, 'readpacket', 'noblock');
    data =  -1;
    if (len > 0)
        data = pnet(sock, 'read');
    end

end

