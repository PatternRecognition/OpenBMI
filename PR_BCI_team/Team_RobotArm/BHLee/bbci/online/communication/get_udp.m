function get_udp
% get_udp - get packages by udp
%
% SYNPOSIS
%    get_udp(hostname, port)
%    data = get_udp([timeout])
%    get_udp('close')
%
% IN
%    hostname: hostname where to connect (either 'localhost' or the
%              name of the local machine)
%        port: portnumber to listen to
%     timeout: how long to wait before returning (in secs)
%       
% OUT
%        data: double array as sent by send_udp
%
% DESCRIPTION
%    get_udp opens a udp socket on the local machine and listens on
%    the given port. By get_udp, one can receive messages sent by
%    send_udp. Messages must be double arrays. You can give an
%    optional timeout argument to get_udp. On timeout, an empty
%    array is returned. Close the port by calling get_udp with some
%    string. 
%
% WRITTEN_BY
%    Mikio Braun
%
% COMPILE_WITH
%    makeget_udp
%
% 2005-08-16
% (c) 2005 by Fraunhofer FIRST
