function acquire_bv
% acquire_bv - get data from the brainserver
%
% SYNPOSIS
%    state = acquire_bv(fs, host)                     [init]
%    state = acquire_bv(fs, host, sampleFilter)       [init]
%    state = acquire_bv(fs, host, bFilter, aFilter)   [init]
%    state = acquire_bv(fs, host, sF, bF, aF)         [init]
%    acquire_bv('close')                              [close]
%    acquire_bv 
%    [data, blockno, markerpos, markertoken, markerdescr] 
%        = acquire_bv(state)        [get data]
%
% ARGUMENTS
%                fs: sampling frequency
%              host: hostname of the brainserver
%  sampleFilter(sF): a filter for the downsampling of the data
%                     the dafault filter will take the newest value of the 
%                     data block
%       bFilter(bF): from an IIR filter the b part
%       aFilter(aF): from an IIR filter the a part
%             state: as returned by acquire_bv
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
%    Conenct to a brainserver and retrieve data. acquire_bv starts
%    a background thread which continuously receives the data from
%    the server and buffers the data.
%       To open a connection, type
%
%           state = acquire_bv(100, 'brainserver')
%        or
%           state = acquire_bv(100, 'brainserver', sampleFilter)
%           state = acquire_bv(100, 'brainserver', bFilter, aFilter)
%           state = acquire_bv(100, 'brainserver', sampleFilter, bFilter, aFilter)
%        if you whish to filter the incoming data from the recorder
%
%    where 100 is the sampling frequency. In order to get data,
%    type at least
%
%           [data, blockno] = acquire_bv(state)
%
%    Or add three more arguments to the left-hand side to also
%    obtain marker information. To close the connection, type
%
%           acquire_bv('close')
%    or
%           acquire_bv
%
% COMPILE WITH
%    makeacquire_bv
%
% AUTHOR
%    Guido Dornhege
%    with a full renovation by Mikio Braun
%    2008/01/29 Max Sagebaum
%                    - update because of changes to acquire.c
%    2008/03/17 Max Sagebaum
%                    - included IIR filter to acquire.c
%    2008/07/01 Max Sagebaum
%                    - added FIR filter to the returned state

% (c) 2005 Fraunhofer FIRST