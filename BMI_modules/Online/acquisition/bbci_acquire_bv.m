function bbci_acquire_bv
% bbci_acquire_bv - get data from the brainserver
%
% SYNPOSIS
%    state = bbci_acquire_bv('init', state)                      [init]
%    state = bbci_acquire_sigserv('init', param1, value1, 
%                                         param2, value2, ...)   [init]
%    
%    [data, markertime, markerdescr, state] 
%        = bbci_acquire_bv(state)                                [get data]
%    bbci_acquire_bv('close')                                    [close]
%    
% ARGUMENTS
%           state: The state object for the initialization
%                  The following fields will be used. If the field is not
%                  present in the struct the default value is used.
%                  Instead of a struct as an argument you can also give the
%                  fields of the struct as a property list.
%                .fs : The sampling frequency
%                      (Default: Original)
%                .host : The hostname for the brainserver
%                        (Default: 127.0.0.1)
%                .filt_b : The b part of the IIR filter.
%                          (Default:  No filter)
%                .filt_a : The a part of the IIR filter.
%                           (Default:  No filter)
%                .filt_subsample: The vector for the subsample filter.
%                           (Default:  Mean value)
%               
%               We will also add the following fields to the state object:
%                .block_no: current block number
%                .chan_sel: channel indices
%                .clab: channel labels
%                .lag: original sampling freq. / sampling freq.
%                .scale: scaling factor
%                .orig_fs: original sampling frequency
%                .reconnect: reconnect to the server on connection loss
%                            (default: true(1))
%                .marker_format: The format of the marker output.
%                         string : e.g. 'R  1', 'S123'
%                         numeric: e.g.    -1 ,   123
%                         (default: numeric);
%
% RETURNS
%          data: [nChans, len] the actual data
%    markertime: [1, nMarkers] marker time
%   markerdescr: {1, nMarkers} marker descriptions
%         state: The updated state object.
%
% DESCRIPTION
%    Conenct to a brainserver and retrieve data. bbci_acquire_bv starts
%    a background thread which continuously receives the data from
%    the server and buffers the data.
%       To open a connection with the default values, type
%
%           params = struct;
%           state = bbci_acquire_bv('init',params)
%       
%        this will open a connection to the server at 127.0.0.1 with a
%        samplingrate of 100 hz and no IIR filters. The state object will
%        be filled with information from the server.
%        In order to get data type at least
%
%           [data] = bbci_acquire_bv(state)
%
%        If you add up to three arguments you will also get 
%        marker information and the state. To close the connection, type
%        
%           bbci_acquire_bv('close')
%
%
% COMPILE WITH
%    make_bbci_acquire_bv
%    in the lib directory

% AUTHOR
%    Max Sagebaum
%    2011/04/19 Max Sagebaum
%                    - Copy from acquire_bv.m
%                    - Changed for new caling conventions.
%    2011/11/17 - Max Sagebaum
%                    - Added the posibility to use a property list as
%                      initialization
%
% (c) 2005 Fraunhofer FIRST