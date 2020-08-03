function send_udp(data)
% send_udp - send data via udp
%
% SYNPOSIS
%    send_udp(hostname, portnumber)
%    send_udp(data)
%    send_udp
%
% IN
%      hostname: name of the target host
%    portnumber: where to send the data
%          data: double array to send
%
% DESCRIPTION
%    send_udp opens an udp socket and sends data to a specific
%    target host. Data must be double arrays. Technically, the size
%    of these packages is limited, and you get an error if the
%    packet size is too large. Close the connection by calling
%    send_udp with no argument at all.
%
% WRITTEN_BY
%    Mikio Braun
%
% COMPILE_WITH
%    makesend_udp.m
%
% 2005-08-16
% (c) 2005 Fraunhofer FIRST
