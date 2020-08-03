function varargout= bbci_acquire_generic(varargin)
%BBCI_ACQUIRE_GENERIC - Online data acquisition compatibility 
%
%Synopsis:
%  STATE= bbci_acquire_generic('init', FS, HOST)
%  STATE= bbci_acquire_generic('init', FS, HOST, FILT_B, FILT_A)
%  [CNTX, MRKTIME, MRKDESC, STATE]= bbci_acquire_generic(STATE)
%  bbci_acquire_generic('close')
% 
%Arguments:
%  FS - Sampling rate
%  HOST - hostname of the machine on which the acquisition software is
%      running.
%  
%Output:
%  STATE - Structure characterizing the incoming signals; fields:
%     'fs', 'clab', and intern stuff
%  CNTX - 'acquired' signals [Time x Channels]
%  The following variables hold the markers that have been 'acquired' within
%  the current block (if any):
%  MRKTIME - DOUBLE: [1 nMarkers] position [msec] within data block
%  MRKDESC - CELL {1 nMarkers} descriptors like 'S 52'
%
%See also:
%  bbci_apply, bbci_acquire_offline

% 02-2011 Benjamin Blankertz


global acquire_func

if length(varargin)>=3,
  if ~ischar(varargin{1}) || ~strcmp(varargin{1}, 'init'),
     error('3 or more input arguments only allowed in init case');
  end
  state= acquire_func(varargin{2:end});
  state.reconnect= 1;
  state.fir_filter= ones(1,state.lag) / state.lag;
  state.fs= state.orig_fs / state.lag;
  output= {state};
elseif length(varargin)~=1,
  error('function takes either 1 or >=3 arguments');
else
  if ischar(varargin{1}),
    if strcmp(varargin{1}, 'close'),
      acquire_func('close');
      return;
    else
      error('unrecognized input argument');
    end
  elseif isstruct(varargin{1}),
    state= varargin{1};
    output= cell(1,5);
    [output{:}]= acquire_func(varargin{:});
    % convert marker positions from [samples] to [msec]
    output{2}= (output{3}+1)/state.fs*1000;
    output{3}= output{4};
    output{4}= state;
  end
end
varargout= output(1:nargout);
