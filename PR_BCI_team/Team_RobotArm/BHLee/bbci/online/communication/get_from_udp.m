function get_from_udp
% get_from_udp - get packages by udp
%
% SYNPOSIS
%    handle = get_from_udp(hostname, port)
%    data = get_from_udp(handle, [timeout])
%    get_from_udp(handle, 'close')
%
% IN
%    hostname: hostname where to connect (either 'localhost' or the
%              name of the local machine)
%        port: portnumber to listen to
%     timeout: how long to wait before returning (in secs)
%      handle: identifier for the connection
%       
% OUT
%        data: double array as sent by send_udp
%      handle: identifier for the connection
%
% DESCRIPTION
%    get_from_udp opens a udp socket on the local machine and listens on
%    the given port. By get_from_udp, one can receive messages sent by
%    send_udp. Messages must be double arrays. You can give an
%    optional timeout argument to get_from_udp. On timeout, an empty
%    array is returned. Close the port by calling get_from_udp with some
%    string. 
%
%    get_from_udp is based on get_udp which could only handle one
%    connection. There is an upper limit on the number of
%    connections, at the moment 64.
%
% COMPILE_WITH
%    make_get_from_udp
%AUTHOR
%    Mikio Braun
%
%    2008/07/01 Max Sagebaum
%                    - added a check if a socket for the hostname and the
%                      port already exsists.
%
% 2006-02-09
% (c) 2006 by Fraunhofer FIRST
