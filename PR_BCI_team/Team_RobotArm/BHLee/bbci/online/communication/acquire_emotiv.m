function acquire_emotiv
% acquire_emotiv - get data from the emotive device
%
% SYNPOSIS
%    state = acquire_emotiv(fs, host)                     [init]
%    state = acquire_emotiv(fs, host, sampleFilter)       [init]
%    state = acquire_emotiv(fs, host, bFilter, aFilter)   [init]
%    state = acquire_emotiv(fs, host, sF, bF, aF)         [init]
%    acquire_emotiv('close')                              [close]
%    acquire_emotiv 
%    [data, blockno, markerpos, markertoken, markerdescr] 
%        = acquire_emotiv(state)        [get data]
%
% ARGUMENTS
%                fs: sampling frequency
%              host: hostname of the Emo Control Panel
%                    if you want to connect directly to the divce type as
%                    host: EmoEngine
%  sampleFilter(sF): a filter for the downsampling of the data
%                     the dafault filter will take the newest value of the 
%                     data block
%       bFilter(bF): from an IIR filter the b part
%       aFilter(aF): from an IIR filter the a part
%             state: as returned by acquire_emotiv
%              .block_no: current block number
%              .chan_sel: channel indices
%                  .clab: channel labels
%                   .lag: original sampling freq. / sampling freq.
%                 .scale: scaling factor
%               .orig_fs: original sampling frequency
%             .reconnect: reconnect to the server on connection loss
%                         (default: false)
%            .fir_filter: the sampleFilter or [0 ... 0 1] as default
%                         you can set a new fir_filter but make shure that
%                         length (newFilter) == lag and
%                         sum(newFilter) == 1
%                         That to say the filter has to be normalized and
%                         the same length as the lag 
%
% RETURNS
%         state: a struct describing the current state of the connection
%          data: [nChans, len] the actual data
%       blockno: the block number
%     markerpos: [1, nMarkers] marker positions
%   markertoken: {1, nMarkers} marker types
%   markerdescr: {1, nMarkers} marker descriptions
%
% DESCRIPTION
%    Conenct to a emotive device and retrieve data.
%       To open a connection, type
%
%           state = acquire_emotiv(128, 'EmoEngine')
%        or
%           state = acquire_emotiv(128, 'EmoEngine', sampleFilter)
%           state = acquire_emotiv(128, 'EmoEngine', bFilter, aFilter)
%           state = acquire_emotiv(128, 'EmoEngine', sampleFilter, bFilter, aFilter)
%        if you whish to filter the incoming data from the recorder
%
%    where 128 is the sampling frequency. In order to get data,
%    type at least
%
%           [data, blockno] = acquire_emotiv(state)
%
%    Or add three more arguments to the left-hand side to also
%    obtain marker information. To close the connection, type
%
%           acquire_emotiv('close')
%    or
%           acquire_emotiv
%
% COMPILE WITH
%    makeacquire_emotiv
%
% AUTHOR
%    Max Sagebaum
%    2011/04/02 Max Sagebaum
%                    - copy from acquire_bv.m

% (c) 2011 Fraunhofer FIRST