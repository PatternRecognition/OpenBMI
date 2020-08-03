function varargout = bbci_acquire_sigserv(varargin)
% bbci_acquire_sigserv - get data from the signalserver
%
% SYNPOSIS
%    state = bbci_acquire_sigserv('init', state)                 [init]
%    state = bbci_acquire_sigserv('init', param1, value1, 
%                                         param2, value2, ...)   [init]
%    [data, markertime, markerdescr, state] 
%        = bbci_acquire_sigserv(state)                           [get data]
%    bbci_acquire_sigserv('close')                               [close]
%    
% ARGUMENTS
%           state: The state object for the initialization
%                  The following fields will be used. If the field is not
%                  present in the struct the default value is used.
%                  Instead of a struct as an argument you can also give the
%                  fields of the struct as a property list.
%                .fs : The sampling frequency
%                      (Default: Original)
%                .host : The hostname for the signalserver
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
%    Conenct to a signalserver and retrieve data. bbci_acquire_sigserv
%    returns one package from the signalserver for every call. There is no
%    internal buffering of the data.
%       To open a connection with the default values, type
%
%           params = struct;
%           state = bbci_acquire_sigserv('init',params)
%       
%        this will open a connection to the server at 127.0.0.1 with a
%        samplingrate of 100 hz and no IIR filters. The state object will
%        be filled with information from the server.
%        In order to get data type at least
%
%           [data] = bbci_acquire_sigserv(state)
%
%        If you add up to three arguments you will also get 
%        marker information and the state. To close the connection, type
%        
%           bbci_acquire_sigserv('close')
%
%
% COMPILE WITH
%    This is a matlab file so no compilation is needed. But this file
%    depends on mexSSClient and acquire_sigserv_filter which must be
%    compiled.

% AUTHOR
%    Max Sagebaum
%   
%    2011/10/20 - Max Sagebaum
%                    - copy from acquire_sigsev.m
%                    - updated for new calling conventions
%    2011/10/27 - Max Sagebaum
%                    - Bugfix: Marker Positions are shifted by -1. So that
%                      they start at zero.
%    2011/11/17 - Max Sagebaum
%                    - Added the posibility to use a property list as
%                      initialization
%    2011/11/21 - Benjamin: corrected dimensions of marker output arguments
%                      and some cosmetic changes


USER_TYPE2_NAME = 'user1'; 

% Comment[BB]: bbci_acquire_* functions should not use persistent variables,
% as it should in principle be possible to acquire signals from two systems
% of the same type independently. This information can be stored in the
% state variable.

% because the markers are a continuous channel we need to store the value
% of the last position in this channel
persistent lastMarkerNumber 
persistent lastBlockNr
persistent lastPackageSize
                            

%% --- Connect to the server
if(1 <= length(varargin) && ischar(varargin{1}) && 1 == strcmp('init', varargin{1}) && 1 == nargout)
  if(2 == length(varargin) && isstruct(varargin{2}))
    outputState = varargin{2};
  elseif( 0 == mod(length(varargin) - 1, 2))
    outputState = propertylist2struct(varargin{2:length(varargin)});
  else
    error('bbci_acquire_sigserv: Init is only allowed with a struct or a property list as an argument.');
  end 
  
  % set the default values we already know
  outputState= set_defaults(outputState, 'host', '127.0.0.1');

  [master_info sig_info ch_info]= mexSSClient(outputState.host, 9000, 'tcp');

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
  
  % set the frequency in the state object
  outputState.orig_fs = master_info(1);
  outputState= set_defaults(outputState, 'fs', outputState.orig_fs);
  
  %calculate the lag
  if( mod(outputState.orig_fs, outputState.fs) ~= 0)
    bbci_acquire_sigserv('close')
    error('bbci_acquire_sigserv: frequency must be a divisor of the original frequency.');
  end
  lag = outputState.orig_fs / outputState.fs;
  
  % set the lag in the state and check for the filters
  outputState.lag = lag;
  outputState.block_no = -1;
  outputState= set_defaults(outputState, ...
                            'filt_b', 1, ...
                            'filt_a', 1, ...
                            'filt_subsample', ones(1, lag) / lag, ...
                            'reconnect', 1, ...
                            'marker_format', 'numeric', ...
                            'chan_sel', 1:nChannels, ...
                            'scale', ones(nChannels,1));
    
  % construct the clab for the state
  clabNumbers = 1:nChannels;
  if(-1 ~= markerChannel && markerChannel <= nChannels)
    % if we have a marker channel omit it in the clabs
    clabNumbers(markerChannel:nChannels) = ...
        clabNumbers(markerChannel:nChannels) + 1;
  end
  outputState.clab = ch_info(clabNumbers, 1);
  
  checkValues(outputState, 1);
  
  %create the filters
  acquire_sigserv_filter('createFilter', outputState.filt_subsample, ...
                         outputState.filt_a, outputState.filt_b, nChannels);
  
  varargout{1} = outputState;
  
  lastMarkerNumber = 0;
  lastBlockNr = -1;
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
      state_temp = bbci_acquire_sigserv('init', state);

      % retry getting data
      [info data] = mexSSClient();
    end
  end
  
%  state.sample_nr= info.sampleNr;
  checkValues(state, 0);
  acquire_sigserv_filter('setFIR', state.filt_subsample);
  
  nChannels = length(state.clab);
  nSamples = size(data{1,2},1);
  nTypes = size(data,1); % Number of different signal types
  outputData = zeros(nChannels,nSamples);
  
  chanPos = 0;
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
      lastMarkerNumber = curData(end);  % save the last value for the
                                        % the next iteration
                                        
      % find the markers (= changes in the marker channel & marker~=0)
      mrkPos= find(diff([lastValue; curData]) & curData);
      nMarkers = length(mrkPos);
      mrkNumber = curData(mrkPos);
      
      % adjust marker position so that they start at an index of zero
      mrkPos = mrkPos-1;
      
      % set the position of the markers to the time in seconds
      mrkPos = mrkPos * 1000 / state.orig_fs;
      
      switch state.marker_format,
       case 'numeric',
        mrkType = mrkNumber;
       case 'string',
        mrkType = cprintf('S%3d', mrkNumber);
       otherwise
        error('bbci_acquire_sigserv: Invalid marker format. Choose "numeric" or "string".');
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
    varargout{2} = mrkPos';
  end
  if(3 <= nargout)
    varargout{3} = mrkType';
  end
  if(4 <= nargout)
    varargout{4} = state;
  end
  
%% --- Close the connection to the server
elseif(1 == length(varargin) && ischar(varargin{1}) && ...
       strcmp('close', varargin{1}))
  mexSSClient('close');
  acquire_sigserv_filter('delete');
else
  error('bbci_acquire_sigserv called with wrong arguments. See the help for further information.'); 
end

return



function checkValues(state, checkall)
%% Currently checks the sizes of the following values
%  filt_a, filt_b, filt_subsample, scale, chan_sel

if checkall,
  if(length(state.filt_a) ~= length(state.filt_b))
    error('bbci_acquire_sigserv: The vectors a and b for the IIR filter must have the same size.');
  end
  if(0 == length(state.filt_a))
    error('bbci_acquire_sigserv: The vectors a and b for the IIR filter must have at least one entry.');
  end
  
  if(state.lag ~= length(state.filt_subsample))
    bbci_acquire_sigserv('close')
    error('bbci_acquire_sigserv: The resample filter must have the size of the lag.');
  end
end

if(state.lag ~= length(state.filt_subsample))
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
