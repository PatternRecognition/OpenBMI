function varargout = acquire_nirs(varargin)
% acquire_nirs - get data from the nirs server
%
% SYNPOSIS
%    state = acquire_nirs(host)                           [init]
%    state = acquire_nirs(host, port)                     [init]
%    state = acquire_nirs(host, port, timeout)            [init]
%    acquire_nirs('close')                                [close]
%    acquire_nirs
%    [data, blockno, markerpos, markertoken, markerdescr]
%        = acquire_nirs(state)                            [get data]
%
% ARGUMENTS
%              port: number of the port for NIRS
%              host: hostname of the brainserver
%           timeout: ms before cutting connection when no frame is received
%             state: as returned by acquire_nirs
%              .block_no: current block number
%              .chan_sel: channel indices
%                  .clab: channel labels
%                   .lag: original sampling freq. / sampling freq.
%                 .scale: scaling factor
%               .orig_fs: original sampling frequency
%             .reconnect: reconnect to the server on connection loss
%                         (default: false)
%
% RETURNS
%         state: a struct describing the current state of the connection
%          data: [nChans, len] the actual data
%       blockno: the block number
%     markerpos: [1, nMarkers] marker positions
%   markertoken: {1, nMarkers} marker types
%   markerdescr: {1, nMarkers} marker descriptions

if (isempty(varargin))||(isstruct(varargin{1})==0)&&(strcmp(varargin{1}, 'close')==0)   %[init]

    %    state = acquire_nirs(host, port, timeout)            [init]

    loadlibrary('TomographyAPI.dll','TomographyAPI.h')

    %Varargin default check
    if (length(varargin) < 1)
        address = 'localhost';
    else
        address = varargin{1};
    end
    if (length(varargin) < 2)
        port = 45342;
    else
        port = varargin{2};
    end
    if (length(varargin) < 3)
        timeout = 500;
     else
        timeout = varargin{3};
    end

    %Initialization
    out = TM_initialize();                                                      %Initialize the API
    [out, addressO] = TM_connect(address, port, timeout);                       %Connect to server
    [out, statusFlagO, sampleRateO] = TM_getStatus();                           %Check the server status
    [out, pNameO] = TM_getName();                                               %Check desired parameters names
    [out, numSourcesO, numDetectorsO, numWavelengthsO] = TM_getChannels();      %Retrieve the number of available channels
    frameSize = numSourcesO*numDetectorsO*numWavelengthsO;
    v = int32([]); pv = libpointer('int32Ptr',v);                               % Set up a pointer

    %Ask the server to start streaming
    [out, sourcesO, detectorsO, wavelengthsO, elementsPerFrameO]  = TM_start(pv, pv, pv, numSourcesO, numDetectorsO, numWavelengthsO, frameSize);


    reqFrames = 1;
    bufferSize = reqFrames * frameSize;

    %Set the initialized state variable and return it
    state = struct('reqFrames',reqFrames, 'timeout',timeout,'bufferSize', bufferSize);
    varargout = {state};
elseif strcmp(varargin(1), 'close')                                  %[close]
    TM_disconnect();
    TM_close();

elseif isstruct(varargin{1})                     %[get data]
    %[data, blockno, markerpos, markertoken, markerdescr]  = acquire_nirs(state)                            [get data]
    reqFrames = varargin{1}.reqFrames;
    timeout = varargin{1}.timeout;
    bufferSize = varargin{1}.bufferSize;
    [out, statusFlagO, sampleRateO] = TM_getStatus();             %Check the server status
    [out, frameCountO, timestampsO, timingBytesO, dataO, dataBufferSizeO]  = TM_getNFrames(reqFrames,timeout,bufferSize);
    if dataBufferSizeO ~= bufferSize                          %Check buffer Size
        bufferSize = dataBufferSizeO;
    end


    %This variables will have to be adapted when we know what are them actually
    %expected to be.
    blockno = 1;
    tB_i = cast(timingBytesO,'uint16');
    tB_bit = dec2bin(tB_i);


    varargout(1) = {dataO};
    varargout(2) = {blockno};
    varargout(3) = {tB_bit};
end
