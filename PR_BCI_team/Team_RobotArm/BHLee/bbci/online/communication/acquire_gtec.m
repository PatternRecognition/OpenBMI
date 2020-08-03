function acquire_gtec
% acquire_gtec - get data from gTec amplifiers
%
% SYNPOSIS
%    state = acquire_gtec(fs, host)                     [init]
%    state = acquire_gtec(fs, host, sampleFilter)       [init]
%    state = acquire_gtec(fs, host, bFilter, aFilter)   [init]
%    state = acquire_gtec(fs, host, sF, bF, aF)         [init]
%    acquire_gtec('close')                              [close]
%    acquire_gtec 
%    [data, blockno, markerpos, markertoken, markerdescr] 
%        = acquire_gtec(state)        [get data]
%
% ARGUMENTS
%                fs: sampling frequency
%              host: hostname of the brainserver
%  sampleFilter(sF): a filter for the downsampling of the data
%                     the dafault filter will take the newest value of the 
%                     data block
%       bFilter(bF): from an IIR filter the b part
%       aFilter(aF): from an IIR filter the a part
%             state: as returned by acquire_gtec
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
%    Conenct to the gTec controllers and retrieve data. acquire_gtec starts
%    a background thread which continuously receives the data from
%    the server and buffers the data.
%       To open a connection, type
%
%           state = acquire_gtec(120, 'brainserver')
%        or
%           state = acquire_gtec(120, 'brainserver', sampleFilter)
%           state = acquire_gtec(120, 'brainserver', bFilter, aFilter)
%           state = acquire_gtec(120, 'brainserver', sampleFilter, bFilter, aFilter)
%        if you whish to filter the incoming data from the recorder
%
%    where 120 is the sampling frequency. In order to get data,
%    type at least
%
%           [data, blockno] = acquire_gtec(state)
%
%    Or add three more arguments to the left-hand side to also
%    obtain marker information. To close the connection, type
%
%           acquire_gtec('close')
%    or
%           acquire_gtec
%
% COMPILE WITH
%    makteacquire_gtec
%
% 2009/01/23 - Max Sagebaum
%               - file created 
%

% (c) TU Berlin

