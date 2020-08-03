function nouzz_connect()

global NOUZZ_UDP_SOCKET

%random port between 5000 and 10000 for sending packets
udp_port = round(5000+rand*5000);

NOUZZ_UDP_SOCKET = pnet('udpsocket', udp_port);
pnet(NOUZZ_UDP_SOCKET, 'udpconnect', '127.0.0.1', 1206);
