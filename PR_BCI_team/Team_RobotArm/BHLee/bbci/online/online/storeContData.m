function cnt = storeContData(mode,varargin)
% storeContData - Ring buffer for continuous EEG data
%
% Synopsis:
%   storeContData(mode,arg1,arg2,...)
%   storeContData('init', nBuffers, nChannels, opt)
%   storeContData('append', whichBuffer, cnt)
%   cnt = storeContData('window', whichBuffer, ilen, timeshift)
%   storeContData('cleanup')
%   
% Arguments:
%   mode: Ring buffer operation. Valid operations are: 'init', 'append',
%       'window'.
%   nBuffers: Number of ring buffers
%   nChannels: scalar or [1 nBuffers]. number of channels in each buffer.
%   whichBuffer: choose the ring buffer to read/write
%   ilen: interval length in milliseconds
%   timeshift: timeshift for end of interval, must be <=0
%
% Description:
%   Syntax for the individual operations:
%   storeContData('init', nBuffers, nChannels, opt)
%       Initialize the ring buffer(s). 
%       nBuffers: number of ring buffers (one for each processing)
%       nChannels: number of channels in each buffer. This can be a
%           scalar or a vector of length nBuffers.
%       opt: Property/Value list or options structure. Recognized options
%           are 'ringBufferSize' (length of the buffers in milliseconds)
%           and 'fs' (sampling frequency)
%           'sloppyfirstappend' (if true, truncate the buffer to fit the
%           number of channels in the first append operation)
%
%   storeContData('append', whichBuffer, cnt)
%       Append data to a chosen ring buffer.
%       whichBuffer: choose the ring buffer to write to
%       cnt: the continuous EEG data to write. If this is longer than the
%           buffer length, only the end of the data will be written.
%
%   cnt = storeContData('window', whichBuffer, ilen)
%   cnt = storeContData('window', whichBuffer, ilen, timeshift)
%       Retrieve a window of data from the end of the buffer.
%       whichBuffer: choose the ring buffer to read from
%       ilen: interval length in milliseconds
%       timeshift: (optional) default 0, if <0: use the interval ending
%           at the given timeshift in the past. timeshift given in
%           milliseconds. 
%       cnt: Windowed EEG data
%
%   storeContData('cleanup')
%       Clean up, i.e., remove the memory-consuming internal
%       variables. After cleanup, a new 'init' operation is required to
%       use this function again.
%   
% Examples:
%   Make a buffer of length 3ms for two channels:
%     storeContData('init', 1, 2, 'ringBufferSize', 3, 'fs', 1000);
%   Store a too long sequence
%     storeContData('append', 1, [1 1; 2 2; 3 3; 4 4]);
%   Retrieving all data in the buffer:
%     storeContData('window', 1, 4)
%   Retrieve the '3' entries:
%     storeContData('window', 1, 1, -1)
%
%
% See also: bbci_bet_apply,persistent
% 

% Author(s): Anton Schwaighofer, Dec 2004
% $Id: storeContData.m,v 1.2 2007/01/11 15:17:28 neuro_cvs Exp $

%error(nargchk(1, inf, nargin));

persistent buffer nBuffers writePtr fillCtr opt bufferLength isInitialized firstAppend

cnt = [];
mode = lower(mode);
if strcmp(mode, 'init'),
  if ~isempty(isInitialized) | isInitialized,
    warning('''init'' call without preceding ''cleanup''. Old data will be lost.');
  end
  nBuffers = varargin{1};
  nChannels = varargin{2};
  if numel(nChannels)==1,
    nChannels = repmat(nChannels, [1 nBuffers]);
  elseif length(nChannels)~=nBuffers,
    error('Number of channels must be a vector of length nBuffers');
  end
  opt = propertylist2struct(varargin{3:end});
  opt = set_defaults(opt, 'ringBufferSize', 15000, 'fs', 100, ...
                          'sloppyfirstappend', 0);
  if numel(opt.ringBufferSize)==1,
    opt.ringBufferSize = repmat(opt.ringBufferSize, [1 nBuffers]);
  elseif length(opt.ringBufferSize)~=nBuffers,
    error('Number of channels must be a vector of length nBuffers');
  end                        
  bufferLength = ceil(opt.ringBufferSize*opt.fs/1000);
  buffer = cell(1, nBuffers);;
  % Pointer where to write next into the buffer
  writePtr = ones([1 nBuffers]);
  % Remember how full the buffer is. This only matters in the beginning,
  % when trying to retrieve too long windows
  fillCtr = zeros([1 nBuffers]);
  for b = 1:nBuffers,
    buffer{b} = zeros([bufferLength(b) nChannels(b)]);
  end
  isInitialized = 1;
  % Store that we will now do the first append operation for each buffer,
  % in case we need to truncate one
  firstAppend = ones([1 nBuffers]);
elseif isempty(isInitialized) | ~isInitialized,
  error('Call to storeContData.m without initialization');
else
  switch lower(mode)
    case 'append'
      b = varargin{1};
      if b<1 | b>nBuffers,
        error('storeContData: Invalid buffer number');
      end
      cnt = varargin{2};
      toWrite = size(cnt,1);
      % Check whether the data is longer than the buffer. If yes: chop off
      if toWrite>bufferLength(b),
        cnt = cnt(end-bufferLength(b)+1:end,:);
        toWrite = size(cnt,1);
        warning('storeContData: Discarded data to fit into buffer');
      end
      % Check whether the number of channels matches:
      if ~opt.sloppyfirstappend | ~firstAppend(b),
        if size(cnt,2)~=size(buffer{b},2),
          error('storeContData: Channel number mismatch');
        end
      else
        % Be sloppy about first append operation. Number of channels can
        % not be easily computed from the processings. At the time of the
        % first append operation, it is clear what the proc do, so we
        % have the number of channels. Chop off the surplus channels
        buffer{b}(:,(size(cnt,2)+1):end) = [];
        buffer{b} = cat(2,buffer{b},zeros(size(buffer{b},1),size(cnt,2)-size(buffer{b},2)));
        firstAppend(b) = 0;
      end
      % Write into buffer. Split this into two parts: toWrite1 is the part
      % of the data that fits into the buffer starting from the current
      % writePtr up to the buffer end, toWrite2 is the part after the
      % wrap-around
      if writePtr(b)+toWrite-1<=bufferLength(b),
        % without wrap-around:
        toWrite1 = toWrite;
      else
        toWrite1 = bufferLength(b)-writePtr(b)+1;
      end
      toWrite2 = toWrite-toWrite1;
      % Write the first part (toWrite1) into the buffer. In most cases,
      % this part will be the only one to write: Avoid doing the indexing
      % in the standard case
      if toWrite2==0,
        buffer{b}(writePtr(b):(writePtr(b)+toWrite1-1),:) = cnt;
      else
        buffer{b}(writePtr(b):(writePtr(b)+toWrite1-1),:) = cnt(1:toWrite1,:);
      end
      % Set new writePtr. writePtr always points to the position where to write next.
      writePtr(b) = writePtr(b)+toWrite1;
      % writePtr can here only reach writePtr==bufferLength(b)+1, not larger
      if writePtr(b)>bufferLength(b),
        writePtr(b) = 1;
      end
      % Do the same with the rest of the data:
      if toWrite2>0,
        buffer{b}(writePtr(b):(writePtr(b)+toWrite2-1),:) = cnt(toWrite1+1:end,:);
        writePtr(b) = writePtr(b)+toWrite2;
        % We can not reach buffer end here, thus no overflow of writePtr
        % possible
      end
      % Remember how full the buffer is.
      fillCtr(b) = min(bufferLength(b), fillCtr(b)+toWrite);
    
    case 'window'
      b = varargin{1};
      if b<1 | b>nBuffers,
        error('storeContData: Invalid buffer number');
      end
      % Get arguments and convert from milliseconds to points
      ilen = varargin{2};
      ilen = ceil(ilen*opt.fs/1000);
      if nargin<4,
        timeshift = 0;
      else
        timeshift = varargin{3};
        timeshift = ceil(timeshift*opt.fs/1000);
      end
      if timeshift>0,
        error('storeContData: Improbability drive not installed. Can not retrieve windows from the future.');
      end
      padWithNaNs = 0;
      if fillCtr(b)<(ilen+abs(timeshift)),
        fprintf('storeContData: Not enough data in buffer, padding with NaNs\n');
        % If not enough data, return all data we have and fill the rest
        % with NaNs
        startPtr = writePtr(b)-fillCtr(b);
        % Remember how much we need to pad: This is from 0 to the actual
        % start of the data
        padWithNaNs = ilen-(fillCtr(b)-abs(timeshift));
        % Pretend that we only have shorter ilen
        ilen = fillCtr(b)-abs(timeshift);
      else
        % Normal case, where we have enough data:
        % Start index to read data
        startPtr = writePtr(b)+timeshift-ilen;
      end
      % Check for underrun (wrap-around)
      if startPtr<1,
        % Allocate memory for result first:
        cnt = zeros([ilen size(buffer{b},2)]);
        % Wrap read pointer around:
        startPtr = bufferLength(b)+startPtr;
        % read that many entries up to the buffer end. With given time
        % shift, the rest up to buffer end must not be fully read, thus
        % restrict to ilen
        toRead = min(ilen, bufferLength(b)-startPtr+1);
        cnt(1:toRead,:) = buffer{b}(startPtr:(startPtr+toRead-1),:);
        % Read the rest up to interval length from the beginning of the
        % buffer
        cnt(toRead+1:end,:) = buffer{b}(1:(ilen-toRead),:);
      else
        % Normal case: read data of length ilen from buffer
        cnt = buffer{b}(startPtr:(startPtr+ilen-1),:);
      end
      % If necessary, pad with NaNs at the beginning:
      if padWithNaNs>0,
        cnt = vertcat(NaN*ones([padWithNaNs size(buffer{b},2)]), cnt);
      end
      
    case 'cleanup'
      buffer = {};
      nBuffers = 0;
      writePtr = [];
      fillCtr = [];
      bufferLength = 0;
      opt = [];
      isInitialized = 0;
      
    otherwise
      error('storeContData: Invalid mode string');
  end
end
