function varargout= bbci_acquire_randomSignals(varargin)
%BBCI_ACQUIRE_RANDOMSIGNALS - Generate random signals
%
%Synopsis:
%  STATE= bbci_acquire_randomSignal('init', <PARAM>)
%  [CNTX, MRKTIME, MRKDESC, STATE]= bbci_acquire_randomSignals(STATE)
%  bbci_acquire_randomSignals('close', STATE)
% 
%Arguments:
%  PARAM - Optional arguments:
%    'marker_mode': [CHAR, default 'global'] specifies how markers should be
%             read:
%                '' means no markers,
%                'global' means checking the global variable ACQ_MARKER, and
%                'pyff_udp' means receiving markers by UDP in the pyff format
%    'fs':    [DOUBLE, default 100] sampling rate
%    'clab':  {CELL of CHAR} channel labels,
%             default {'F3','Fz','F4','C3','Cz','C4','P3','Pz','P4'}.
%    'amp': amplitude of the signals (factor for 'randn')
%    
%Output:
%  STATE - Structure characterizing the incoming signals; fields:
%     'fs', 'clab', and intern stuff
%  CNTX - 'acquired' signals [Time x Channels]
%  The following variables hold the markers that have been 'acquired' within
%  the current block (if any).
%  MRKTIME - DOUBLE: [1 nMarkers] position [msec] within data block.
%  MRKDESC - DOUBLE: [1 nMarkers] corresponding marker values

% 02-2012 Benjamin Blankertz


% This is just for opt.marker_mode='global':
global ACQ_MARKER

if isequal(varargin{1}, 'init'),
  opt= varargin{2};
  state= ...
      set_defaults(opt, ...
                   'fs', 100, ...
                   'clab', {'F3','Fz','F4','C3','Cz','C4','P3','Pz','P4'}, ...
                   'blocksize', 40, ...
                   'amp', 30, ...
                   'marker_mode', 'global');
  state.nChannels= length(state.clab);
  state.blocksize_sa= ceil(state.blocksize*state.fs/1000);
  state.blocksize= state.blocksize_sa/state.fs*1000;
  output= {state};
  switch(state.marker_mode),
   case 'pyff_udp',
    state.socket= open_udp_socket();
  end
elseif isequal(varargin{1}, 'close'),
  state= varargin{2};
  switch(state.marker_mode),
   case 'pyff_udp',
    close_udp_socket(state.socket);
  end
  return
elseif length(varargin)~=1,
  error('Except for INIT/CLOSE case, only one input argument expected');
else
  if isstruct(varargin{1}),
    state= varargin{1};
    cntx= state.amp*randn(state.blocksize_sa, state.nChannels);
    switch(state.marker_mode),
     case '',
      mrkTime= [];
      mrkDesc= [];
     case 'pyff_udp',
      packet= receive_udp(sock);
      if ~isempty(packet),
        % we don't know the marker position within the block -> set randomly
        mrkTime= ceil(state.blocksize_sa*rand)/state.fs*1000;
        mrkDesc= str2int(packet);  % -> check format
      else
        mrkTime= [];
        mrkDesc= [];
      end
     case 'global',
      if ~isempty(ACQ_MARKER),
        mrkTime= ceil(state.blocksize_sa*rand)/state.fs*1000;
        mrkDesc= ACQ_MARKER;
        ACQ_MARKER= [];
      else
        mrkTime= [];
        mrkDesc= [];
      end
     otherwise
      error('unknown marker_mode: %s', state.marker_mode);
    end
    output= {cntx, mrkTime, mrkDesc, state};
  else
    error('unrecognized input argument');
  end
end
varargout= output(1:nargout);
