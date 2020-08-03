% test_send_to_udp - test the new send_to_udp, get_from_udp functions
%
% DESCRIPTION
%	opens some ports and sends some data. this function sends 
%	data to three different ports. You should then call
%	test_get_from_udp simulataneously to receive the data. Actually, you
%	should call test_get_from_udp first. :)
%
% AUTHOR
%	Mikio Braun

h1 = send_to_udp('localhost', 4751)
h2 = send_to_udp('localhost', 4752)
h3 = send_to_udp('localhost', 4753)

send_to_udp(h1, [1,2,3,4,5])
send_to_udp(h2, [6,7,8,9,10])
send_to_udp(h3, [11,12,13,14,15]);

send_to_udp(h1, 'close')
send_to_udp(h2, 'close')
send_to_udp(h3, 'close')

send_to_udp(h1, [1,2,3,4,5]) % not connected!

