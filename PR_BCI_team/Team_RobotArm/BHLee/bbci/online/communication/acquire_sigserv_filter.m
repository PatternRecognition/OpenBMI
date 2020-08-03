function acquire_sigserv_filter
%  acquire_sigserv_filter.m
%
% SYNOPSIS 
%   acquire_sigserv_filter('createFilter', firFilter, aFilt, bFilt,
%                           nChannels)                        [init]  
%   acquire_sigserv_filter('setFIR', firFilter)               [reinit]
%   acquire_sigserv_filter('delete')                          [cleanup]
%   filData = acquire_sigserv_filter('filter',data, state)    [data]
%
% ARGUMENTS
%    firFilter: The vector for the resampling filter
%        aFilt: The a vector for the IIR filter
%        bFilt: The b vector for the IIR filter
%    nChannels: The number of channels we want to filter
%         data: The data we want to filter.It must have the dimension
%               [nChannels x nSamples]
%        state: The state structure from acquire_sigserv
%               (The fields 'lag', 'chan_sel' and 'scal' are used.
%
% RETURNS
%     filtData: The output of the filtered data
%
% DESCRIPTION
%   This function is used by acquire_sigserv to filter the incoming data in
%   the same way as acquire_bv, ...
%   The filters are used by first initializing them with 'createFilter'.
%   Then you can filter your data with 'filter'. When you are
%   finisched you can clean up everything with 'delete'.
% 
% COMPILE
%   make_acquire_sigserv_filter
%   
%  2010/08/30 - Max Sagebaum 
%                 - file created
%   
%  (c) Fraunhofer FIRST.IDA 2010