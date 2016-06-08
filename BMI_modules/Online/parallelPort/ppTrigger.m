function ppTrigger(value,raiseTime, delayTime)
% function - ppTrigger
%
% Send a value to the parallel port
%
% Synopsis:
%   ppTrigger(value)
%   ppTrigger(value, raiseTime)
%   ppTrigger(value, raiseTime, delayTime)
%
% Description:
%   The function uses ppWrite to write a value on the parallel port if 
%   the global variable IO_ADDR is set. If the global variable 
%   NOUZZ_UDP_SOCKET is set the values are also send via udp.
%
%   Currently the delay is not implemented for nouzz.
%
% Input: 
%       value: The value you want write on the parallel port
%   raiseTime:[Optional] The time in ms how long the value should stay on
%             the parallel port. (Default: 10ms)
%   delayTime:[Optional] The time in ms how long the value should be
%              delayed. (Default: 0ms)

% Author ??? (Max Sagebaum)
%   2011/11/16 - Max Sagebaum
%                - Added documentation
%                - Added delayTime argument

global IO_ADDR
global NOUZZ_UDP_SOCKET
%global ETH_MARKERS

if ~isempty(IO_ADDR)
    if exist('delayTime','var')
      ppWrite(IO_ADDR, value, raiseTime, delayTime);
    elseif exist('raiseTime','var')
      ppWrite(IO_ADDR, value, raiseTime);
    else
      ppWrite(IO_ADDR, value);
    end
end

if ~isempty(NOUZZ_UDP_SOCKET)
    if exist('delayTime','var')
        if 0 ~= delayTime
            warning('ppTrigger: delay is not implemented for nouzz');
        end
    end
    pnet(NOUZZ_UDP_SOCKET, 'write', [uint8('M') typecast(uint16(value),'uint8')]);
    pnet(NOUZZ_UDP_SOCKET, 'writepacket');
end

%if (~isempty(ETH_MARKERS))
%    fprintf('redirecting trigger eth: %d\n', value);
 %   rawethmex('sendto', 1, sprintf('S%3d',value), 3); 
%end

