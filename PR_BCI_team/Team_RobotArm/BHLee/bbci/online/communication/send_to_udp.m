function h = send_to_udp(a, b)
% send_to_udp - send data via udp
%
% SYNPOSIS
%    handle = send_to_udp(hostname, portnumber)
%    send_to_udp(handle, data)
%    send_to_udp(handle, 'close')
%
% IN
%      hostname: name of the target host
%    portnumber: where to send the data
%          data: double array to send
%        handle: handle returned by the opening call
%
% OUT
%        handle: connection identifier
%
% DESCRIPTION
%    send_to_udp opens an udp socket and sends data to a specific
%    target host. Data must be double arrays. Technically, the size
%    of these packages is limited, and you get an error if the
%    packet size is too large. Close the connection by calling
%    send_to_udp with handler and string 'close'
%
%    send_to_udp is an extension of send_udp. send_to_udp can
%    handle several hosts and portnumbers.
%
%    There is a maximum number of possible connections, which is
%    larger than 64.
%
% WRITTEN_BY
%    Mikio Braun
%
% COMPILE_WITH
%    make_send_to_udp.m
%
% 2006-02-09
% (c) 2006 Fraunhofer FIRST
