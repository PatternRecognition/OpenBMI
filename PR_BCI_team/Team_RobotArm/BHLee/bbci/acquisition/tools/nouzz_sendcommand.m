function nouzz_sendcommand(fcn, varargin)
%NOUZZ_SENDCOMMAND - Control the Nouzz Recorder software
%
%Synopsis:
% nouzz_sendcommand(FCN, <ARGS>)
%
%Arguements:
% FCN: Name of the function to be executed:
%     'startrecording' - Start EEG recording; ARG: name of the file with
%        full path, without extension.
%     'stoprecording' - Stops the recording.
%     To send markers, use ppTrigger

% marton@cs.tu-berlin.de, Nov-2010

global NOUZZ_UDP_SOCKET

if isempty(NOUZZ_UDP_SOCKET)
    return;
end

if strcmp(fcn, 'startrecording')
    if numel(varargin)<1
        error('Please specify file name');
    end
    cmd = sprintf('B%s',varargin{1});
elseif strcmp(fcn, 'stoprecording')
    cmd = 'E';
end

pnet(NOUZZ_UDP_SOCKET, 'write', cmd);
pnet(NOUZZ_UDP_SOCKET, 'writepacket');
