function varargout = acquire_sigserv(varargin)
% acquire_sigserv - get data from the signalserver
%
% SYNPOSIS
%    state = acquire_sigserv(fs, host)                     [init]
%    state = acquire_sigserv(fs, host, sampleFilter)       [init]
%    state = acquire_sigserv(fs, host, bFilter, aFilter)   [init]
%    state = acquire_sigserv(fs, host, sF, bF, aF)         [init]
%    acquire_sigserv('close')                              [close]
%    [data, blockno, markerpos, markertoken, markerdescr] 
%        = acquire_sigserv(state)                          [get data]
%
% ARGUMENTS
%                fs: sampling frequency
%              host: hostname of the signalserver
%  sampleFilter(sF): The vector for the resampling filter
%       bFilter(bF): The a vector for the IIR filter
%       aFilter(aF): The b vector for the IIR filter
%             state: as returned by acquire_sigserv
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
%    Conenct to a signalserver and retrieve data. acquire_sigserv calls
%    mexSSClient an merges the data into one big datastructure
%       To open a connection, type
%
%           state = acquire_sigserv(100, '10.10.10.1')
%        or
%           state = acquire_sigserv(100, '10.10.10.1', sampleFilter)
%           state = acquire_sigserv(100, '10.10.10.1', bFilter, aFilter)
%           state = acquire_sigserv(100, '10.10.10.1', sampleFilter, bFilter, aFilter)
%        if you whish to filter the incoming data from the recorder
%
%    where 100 is the sampling frequency. In order to get data,
%    type at least
%
%           [data, blockno] = acquire_sigserv(state)
%
%    Or add three more arguments to the left-hand side to also
%    obtain marker information. To close the connection, type
%
%           acquire_sigserv('close')
%    or
%           acquire_sigserv

% AUTHOR
%    Max Sagebaum
%   
%    2010/04/12 - Max Sagebaum
%                    - file created
%    2010/08/30 - Max Sagebaum
%                  - added filtering support
%    2010/09/13 - Max Sagebaum
%                  - The downsampling wasn't applied to the sample number.
%    2010/09/21 - Max Sagebaum
%                  - The sample number from the server is not the sample
%                    number we need. Added logic to compute the sample
%                    number.
%    2010/10/08 - Max Sagebaum
%                  - Added logic which reconnects to the server.
%    2011/10/27 - Max Sagebaum
%                  - Added a check for some values in the state structure
%                  - FIR Filter is now set from the value in the state
%                    structure

% (c) 2010 Fraunhofer FIRST
  USER_TYPE2_NAME = 'user1'; 
  
  % because the markers are a continuous channel we need to store the value
  % of the last position in this channel
  persistent lastMarkerNumber 
  persistent lastBlockNr
  persistent lastPackageSize
                              
  
%% --- Connect to the server
  if(2 <= length(varargin))
    server = varargin{2};
    [master_info sig_info ch_info] = mexSSClient(server,9000,'tcp');

    % Check if we have a marker channel and count the number of channels
    nChannels = 0;
    markerChannel = -1;
    for ch = 1:size(sig_info,1)
      if(~strcmp(USER_TYPE2_NAME, sig_info{ch,2}))
        nChannels = nChannels + sig_info{ch,4};
      else
        markerChannel = nChannels + 1;
      end
    end
    %calculate the lag
    fs = varargin{1};
    orig_fs = master_info(1);
    if( mod(orig_fs,fs) ~= 0)
      acquire_sigserv('close')
      error('acquire_sigserv: frequency must be a divisor of the original frequency.');
    end
    lag = orig_fs / fs;
    
    %set the default filters
    resampleFilter = ones(1,lag)/lag; % set default to subsample to mean
    iirFilterA = 1;
    iirFilterB = 1;
    %check for the filter arguments
    if(3 == length(varargin))
      resampleFilter = varargin{3};
    elseif(4 == length(varargin))
      iirFilterA = varargin{4};
      iirFilterB = varargin{3};
    elseif(5 == length(varargin))
      resampleFilter = varargin{3};
      iirFilterA = varargin{5};
      iirFilterB = varargin{4};
    end
    
    %create the filters
    acquire_sigserv_filter('createFilter', resampleFilter, iirFilterA, iirFilterB,nChannels);
    
    % construct the clab for the state
    clabNumbers = 1:nChannels;
    if(-1 ~= markerChannel && markerChannel <= nChannels)
      % if we have a marker channel omit it in the clabs
      clabNumbers(markerChannel:nChannels) = clabNumbers(markerChannel:nChannels) + 1;
    end
    clab = ch_info(clabNumbers,1);
    
    %% Create the info structure
    state = struct();
    state.block_no = 0;
    state.chan_sel = 1:nChannels;
    state.clab = clab;
    state.lag = lag;
    state.scale = ones(nChannels,1);
    state.fs = fs;
    state.orig_fs = orig_fs;
    state.reconnect = 0;
    state.fir_filter = resampleFilter;
    state.hostname = server;
    state.iir_a_filter = iirFilterA;
    state.iir_b_filter = iirFilterB;
    
    varargout{1} = state;
    
    lastMarkerNumber = 0;
    lastBlockNr = 0;
    lastPackageSize = 1;
%% --- Receive data from the server
  elseif(1 == length(varargin) && isstruct(varargin{1}) == 1)
    state = varargin{1};
    
    % get the data
     try 
        [info data] = mexSSClient();
     catch
      if(state.reconnect ~= 1)
        rethrow(EX)
      else
        warning('acquire_sigserv: Error getting data. Reconnecting.');
        % we do not want to save the state vairable
        state_temp = acquire_sigserv(state.fs, state.hostname, state.fir_filter, state.iir_b_filter, state.iir_a_filter);
        
        % retry getting data
        [info data] = mexSSClient();
      end
     end
    
    checkValues(state);
    
    acquire_sigserv_filter('setFIR', state.fir_filter);
    
    nChannels = length(state.clab);
    nSamples = size(data{1,2},1);
    
    nTypes = size(data,1); % Number of different signal types
    
    outputData = zeros(nChannels,nSamples);
    
    chanPos = 0;
    mrkPos = zeros(1,0);
    mrkType = cell(1,0);
    mrkDesc = cell(1,0);
    for type = 1:nTypes
      curData = data{type,2};
      
      % Check if the data block contains signal data or markers
      if(~strcmp(USER_TYPE2_NAME, data{type,1}))
        curChannels = size(curData,2);

        outputData((1:curChannels) + chanPos,:) = curData';
        chanPos = chanPos + curChannels;
      else 
        % marker channel
        % First set every value to zero which whould result in a stream of 
        % markers. We only want to know if the marker value has changed
        lastValue = lastMarkerNumber;
        lastMarkerNumber = curData(length(curData)); % save the last value for the
                                                     % the next iteration
        for i = 1:length(curData)
          if(curData(i) == lastValue)
            curData(i) = 0;
          else
            lastValue = curData(i);
          end
        end
        
        % find the markers
        mrkPos = find(curData ~= 0);
        nMarkers = length(mrkPos);
        mrkNumber = curData(mrkPos);

        % apply the resampling for the markers
        mrkPos = floor((mrkPos-1) / state.lag);
        
        mrkType = cell(nMarkers,1);
        mrkDesc = cell(nMarkers,1);
        for i=1:length(mrkPos)
          strNumber = num2str(mrkNumber(i));
          spaces = '';
          switch(length(strNumber))
            case 0
              spaces = '   ';
            case 1
              spaces = '  ';
            case 2 
              spaces = ' ';
          end
              
          mrkType{i} = ['S' spaces strNumber];
          mrkDesc{i} = 'Stimulus';
        end
      end
    end

    %filter the data
    outputData = acquire_sigserv_filter('filter', outputData, state);
    
    % update the sample number (block_no is actually the sample number)
    lastBlockNr = lastBlockNr + lastPackageSize;
    lastPackageSize = size(outputData,1);
    state.block_no = lastBlockNr;
    
    varargout{1} = outputData;
    if(2 <= nargout)
      varargout{2} = state.block_no;
    end
    if(3 <= nargout)
      varargout{3} = mrkPos;
    end
    if(4 <= nargout)
      varargout{4} = mrkType;
    end
    if(5 <= nargout)
      varargout{5} = mrkDesc;
    end
%% --- Close the connection to the server
  elseif( 0 == length(varargin) || (1 == length(varargin) && 1 == ischar(varargin{1})))
    mexSSClient('close');
    acquire_sigserv_filter('delete');
  end
end

%% Currently checks the sizes of the following values
%  scale
%  chan_sel
%  filt_subsample
function checkValues(state)
  if(state.lag ~= length(state.fir_filter))
    bbci_acquire_sigserv('close')
    error('bbci_acquire_sigserv: The resample filter must have the size of the lag.');
  end
  
  nChannels = length(state.clab);
  if(nChannels ~= length(state.scale))
    bbci_acquire_sigserv('close')
    error('bbci_acquire_sigserv: The scaling factors must have the size of the channels.');
  end
  
  for i = 1:state.chan_sel
    if(state.chan_sel(i) > nChannels || state.chan_sel(i) < 1)
      bbci_acquire_sigserv('close')
      error('bbci_acquire_sigserv: The channels in the chan_sel must be between 1 and  the number of channels.');
    end
  end
end