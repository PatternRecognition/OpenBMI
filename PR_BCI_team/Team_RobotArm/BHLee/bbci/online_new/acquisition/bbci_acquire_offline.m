function varargout= bbci_acquire_offline(varargin)
%BBCI_ACQUIRE_OFFLINE - Simulating online acquisition by reading from a file
%
%Synopsis:
%  STATE= bbci_acquire_offline('init', CNT, MRK)
%  STATE= bbci_acquire_offline('init', CNT, MRK, <OPT>)
%  [CNTX, MRKTIME, MRKDESC, STATE]= bbci_acquire_offline(STATE)
%  bbci_acquire_offline('close')
% 
%Arguments:
%  CNT - Structure of continuous data, see eegfile_readBV
%  MRK - Structure of markers, see eegfile_readBVmarkers
%  OPT - Struct or property/value list of optinal properties:
%    .blocksize - [INT] Size of blocks (msec) in which data should be
%          processed, default 40.
%    .realtime - [BOOL] If true, this functions wait such that data blocks
%          are returned approximately that the speed specified by 'blocksize'.
%
%Output:
%  STATE - Structure characterizing the incoming signals; fields:
%     'fs', 'clab', and intern stuff
%  CNTX - 'acquired' signals [Time x Channels]
%  The following variables hold the markers that have been 'acquired' within
%  the current block (if any).
%  MRKTIME - DOUBLE: [1 nMarkers] position [msec] within data block
%  MRKDESC - CELL {1 nMarkers} descriptors like 'S 52'
%
%See also:
%  bbci_apply, bbci_acquire_bv

% 02-2011 Benjamin Blankertz


if isequal(varargin{1}, 'init'),
  if nargin<3,
    error('CNT and MRK must be provided as input arguments');
  end
  state= propertylist2struct(varargin{4:end});
  state= set_defaults(state, ...
                      'blocksize', 40, ...
                      'realtime', 0);
  if state.realtime,
    waitForSync;
  end
  [cnt, mrk]= varargin{2:3};
  if cnt.fs~=mrk.fs,
    error('sampling rates in CNT and MRK not consistent');
  end
  state.lag= 1;
  state.orig_fs= cnt.fs;
  state.fs= cnt.fs;
  state.cnt_step= round(state.blocksize/1000*cnt.fs);
  state.cnt_idx= 1:state.cnt_step;
  state.clab= cnt.clab;
  state.cnt= cnt;
  if ~isfield(mrk, 'desc'),
    mrk.desc= mrk.toe;
  end
  state.mrk= mrk;
  output= {state};
elseif length(varargin)~=1,
  error('Except for INIT case, only one input argument expected');
else
  state= varargin{1};
  if isequal(varargin{1}, 'close'),
    state= struct('cnt_idx', []);      % it is not really to close
    return
  elseif isstruct(varargin{1}),
    if isempty(state.cnt_idx),
      error('file is closed');
    end
    if state.cnt_idx(end) > size(state.cnt.x,1),
      state.running= false;
      output= {[], [], [], state};
      varargout= output(1:nargout);
      return;
    end
    
    cntx= state.cnt.x(state.cnt_idx, :);
    mrk_idx= find(state.mrk.pos>=state.cnt_idx(1) & ...
                  state.mrk.pos<=state.cnt_idx(end));
    mrkTime= (state.mrk.pos(mrk_idx)-state.cnt_idx(1)+1)/state.fs*1000;
    mrkDesc= state.mrk.desc(mrk_idx);         
    state.cnt_idx= state.cnt_idx + state.cnt_step;
    output= {cntx, mrkTime, mrkDesc, state};
    if state.realtime,
      waitForSync(state.blocksize);
    end
  else
    error('unrecognized input argument');
  end
end
varargout= output(1:nargout);
