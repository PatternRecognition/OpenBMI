function varargout = bbci_acquire_nirx(varargin)
% BBCI_ACQUIRE_NIRX - Get data from the NIRx system
%
% SYNPOSIS
%    STATE = bbci_acquire_nirx('init', STATE)                 [init]
%    [DATA, MARKERTIME, MARKERDESC, STATE] 
%        = bbci_acquire_nirx(state)                           [get data]
%    bbci_acquire_nirx('close')                               [close]
%    bbci_acquire_nirx('close', DUMMY)                        [close]
%    
% ARGUMENTS
%           state: The state object for the initialization
%                  The following fields will be used. If the field is not
%                  present in the struct the default value is used.
%                  Instead of a struct as an argument you can also give the
%                  fields of the struct as a property list.
%                .host : The hostname for the signalserver
%                        (CHAR Default: '127.0.0.1')
%                .port : Port number (default 45342)
%                .timeout: ms before cutting connection when no frame is received
%                          (default 1000)
%                .nFrames    : how many frames should be fetched upon each
%                             call (default 1)
%                .buffersize: how many frames should be buffered (default state.frames)
%                .sources: vector of source indices that should be streamed
%                          (default [] yields all sources)
%                .detectors: vector of detector indices that should be streamed
%                          (default [] yields all detectors)
%                .wavelengths: vector of wavelength indices that should be streamed
%                          (default [] yields all wavelengths)
%                (NOT USED) .filt_b : The b part of the IIR filter.
%                          (Default:  No filter)
%                (NOT USED) .filt_a : The a part of the IIR filter.
%                           (Default:  No filter)
%                .LB      : if 1 Lambert-Beer is applied, converts the two
%                           wavelengths into oxy and deoxy (default 0)
%               
%           Further fields acquired from the NIRx API are added, 
%           all starting with .nirx_*
%            numSources         number of sources
%            numDetectors       number of detectors
%            version            version of API
%            statusFlag         flag indicating status of connection with NIRx server
%            status             string specifying the status
%                
%           Other fields:
%                .marker_format: The format of the marker output.
%                         string : e.g. 'R  1', 'S123'
%                         numeric: e.g.    -1 ,   123
%                         (default: numeric);
%
% RETURNS
%          data: [nChans, len] the actual data. The data format is 
%                sources * detectors * wavelengths.
%    markertime: [1, nMarkers] marker time in milliseconds
%   markerdescr: {1, nMarkers} marker descriptions
%         state: The updated state object.
%
% DESCRIPTION
%    Connects to the NIRx system. The NIRx dynamic link library 
%    Tomography.dll needs to be installed in the import folder.
%
%    Server side: The NIRx NIRStar software should be set to stream the 
%    data. Go to Hardware Configuration -> Data Streaming:
%    * Stream Data should be on
%    * Buffer Depth is the number of buffered frames (ie full NIRS sweeps)
%    and should be more than the amount of frames you typically pick up
%    (default 2).
%
% See also: BBCI_ACQUIRE_SIGSERV

% AUTHOR
%    Matthias Treder 12-2011


global BCI_DIR


%% Initialize
if ischar(varargin{1}) && strcmp(varargin{1},'init') % && isstruct(varargin{2}) && nargin==2

  v = int32([]);
  pv = libpointer('int32Ptr',v);    % Make empty pointer

  if nargin<2
    state = {};
  else
    state = propertylist2struct(varargin{2:end});
  end
   
  [state,isdefault] = set_defaults(state, ...
    'host','localhost',  ...
    'port',45342, ...
    'timeout', 5000, ...
	  'marker_format', 'numeric', ...
    'sources',pv, ...
    'detectors',pv, ...
    'wavelengths',pv, ...
    'nFrames',1, ...
    'buffersize', 2, ...
    'LB', 0 ...
    );

 
  NIRX_PATH = [BCI_DIR 'import/TomographyMATLAB_API/'];
  if ~appendpathifexists(NIRX_PATH)
    error('Path to NIRx DLL (%s) not found',NIRX_PATH)
  end

  LIB = 'TomographyAPI';
  if ~libisloaded(LIB), loadlibrary([LIB '.dll'],[LIB '.h']);   end
  pause(.5)
  
  out = calllib(LIB,'tsdk_initialize');
  checkIfNirxError(out); pause(.1)

  [out, addressO] = calllib(LIB,'tsdk_connect',state.host, state.port, state.timeout);
  checkIfNirxError(out);
  
  % NIRx info
  nirxinfo = struct();
  nirxinfo.version = calllib(LIB,'tsdk_util_getAPIVersion',struct());
  fprintf('Loaded NIRx TomographyAPI version %d.%d.%d.%d\n',cell2mat(struct2cell(nirxinfo.version)))
  [out, statusFlag, fs] = calllib(LIB,'tsdk_getStatus', 0, 0);
  % Get maximum number of sources/detectors/channels available
  [out, maxSources, maxDetectors, maxWavelengths] = calllib(LIB,'tsdk_getChannels', 0, 0, 0);
  checkIfNirxError(out);

  % Store info in state
  state.fs = fs;
  nirxinfo.maxSources = maxSources;
  nirxinfo.maxDetectors = maxDetectors;
  nirxinfo.maxWavelengths = maxWavelengths;
	
  if isdefault.sources,  nirxinfo.nSources = maxSources;
  else                   nirxinfo.nSources = numel(state.sources); end
  if isdefault.detectors,  nirxinfo.nDetectors = maxDetectors;
  else                   nirxinfo.nDetectors = numel(state.detectors); end
  if isdefault.wavelengths,  nirxinfo.nWavelengths = maxWavelengths;
  else                   nirxinfo.nWavelengths = numel(state.wavelengths); end
  
  nirxinfo.nChans = nirxinfo.nSources * nirxinfo.nDetectors;
  state.buffersize = nirxinfo.nChans * 2; 
  % Start streaming
  elementsPerFrame = 0;
  pause(.5)
   [out, sourcesO, detectorsO, wavelengthsO, elementsPerFrame] = ...
      calllib(LIB,'tsdk_start', ...
      state.sources, state.detectors, state.wavelengths, ...
      1,64, 2, ...
      state.buffersize, elementsPerFrame);
  checkIfNirxError(out);
  
%   [out, statusFlag] = calllib(LIB,'tsdk_getStatus', 0, 0);
%   nirxinfo.statusFlag = statusFlag;
%   nirxinfo.status = flag2status(statusFlag);
  nirxinfo.elementsPerFrame = elementsPerFrame;
  state.nirxinfo = nirxinfo;




  varargout{1} = state;
  
%% Acquire data
elseif isstruct(varargin{1}) && nargin==1
  state = varargin{1};  
  
  frameCount = 0;
  timestamps = 0;
  timingBytes = '';
  data = zeros(1,state.nirxinfo.elementsPerFrame);
  
  [out, frameCount, timestamps, timingBytes, data]  =  ...
    calllib('TomographyAPI','tsdk_getNFrames', ...
    state.nFrames, state.timeout, frameCount, timestamps, timingBytes, ...
    data, state.nirxinfo.elementsPerFrame);
  
  % 
  if frameCount>1
    warning('No support for framecount > 1 so far')
  end
  
  % Check for markers
  if any(~isempty(timingBytes))
    if strcmp(state.marker_format,'numeric')
      markerdesc = cast(timingBytes,'uint16');
    else
      markerdesc = sprintf('S%3d', timingBytes);
    end
    markertime = 1/state.fs;   % fixed since we have only one sample
  else
    markertime = [];
    markerdesc = [];
  end

  varargout{1}= data;
  varargout{2}= markertime;
  varargout{3}= markerdesc;
  varargout{4}= state;

%% Close connection
elseif ischar(varargin{1}) && strcmp(varargin{1},'close') && nargin<=2,

  calllib('TomographyAPI','tsdk_stop');  % Ask server to stop sending data
  calllib('TomographyAPI','tsdk_disconnect'); % Disconnect NIRx server
  calllib('TomographyAPI','tsdk_close'); % Cloes the driver and frees memory
  unloadlibrary('TomographyAPI'); % Unload DLL
  
else
  error('Could not parse call with %d arguments; wrong number or type of arguments',nargin)
end


%% Misc

function checkIfNirxError(err)
  % Checks if return argument refers to an error. If yes, gives the error
  % output.
	if err~=0
    calllib('TomographyAPI','tsdk_util_getErrorMsg',err,'',200)
  end
end

function s = flag2status(flag)
  % Returns a string describing the status(es) associated with a particular
  % flag. Strings taken from TomographyAPI.h
  b = dec2bin(flag);
  states = {'ACQUIRING' 'OVERFLOW' 'TRANSMITTING' 'CONNECTED' 'CLIENT_DATA_OVERFLOW'};
  s = {states{strfind(b,'1')}};
% More efficient (bb):
% s= states(find(bitget(flag,5:-1:1)));
end

end



