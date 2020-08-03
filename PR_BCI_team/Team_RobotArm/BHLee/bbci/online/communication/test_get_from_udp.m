% test_get_from_udp
%
% DESCRIPTION
%	Call this function first, then call test_send_to_udp from a
%	parallel matlab session. You should see the numbers 1:15
%

h1 = get_from_udp('localhost', 4751)
h2 = get_from_udp('localhost', 4752)
h3 = get_from_udp('localhost', 4753)

data = get_from_udp(h1)
data = get_from_udp(h2)
data = get_from_udp(h3)

get_from_udp(h1, 'close')
get_from_udp(h2, 'close')
get_from_udp(h3, 'close')
