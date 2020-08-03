function hdr= nirsfile_readHeader(file, varargin)
% NIRSFILE_READHEADER - Read NIRS Header from single or multiple header
%                       files.
%
% Synopsis:
%   HDR= nirs_readHeader(HDRNAME, 'Property1',Value1, ...)
%
% Arguments:
%   HDRNAME: name of header file (no extension),
%            relative to NIRS_RAW_DIR (if exists) or EEG_RAW_DIR 
%            unless beginning with '/'.
%            HDRNAME may also contain '*' as wildcard or be a cell
%            array of strings
%
% Properties:
%   'path' : directory with raw NIRS data (default NIRS_RAW_DIR (if exists)
%           or EEG_RAW_DIR). If data is file contains an absolute path,
%           path is ignored.
%   'system': NIRS system used (default 'nirx')
%   'headerExt': file extension for header (by default determined
%                automatically based on the system, eg '.hdr')
%
% Returns:
%   HDR: header structure:
%   .fs    - sampling frequency
%
%   nirx fields:
%   .Gains - gain setting of photon counter (smth like impedances in EEG).
%            Possible values 1-10, the higher the better, 6-8 is realistic.
%            Since always two sources are 'on' at the same time, there's
%            gain settings for only half of the sources.
%   .Sources/.Detectors - number of sources and detectors
%   .SDkey - coupling between source (1st col) and detector (2nd col). The
%            order corresponds to the order of source-detector channels in
%            the x (data) field.
%
% Based on eegfile_readBVheader
% See also: nirsfile_* nirs_*

% matthias.treder@tu-berlin.de 2011

%% Set file path
if ~iscell(file) && isabsolutepath(file),
  nirsdir = [];
elseif ~isempty(whos('global','NIRS_RAW_DIR'))
  global NIRS_RAW_DIR
  nirsdir = NIRS_RAW_DIR;
else
  global EEG_RAW_DIR
  nirsdir = EEG_RAW_DIR;
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'headerExt',[],...
                 'system','nirx',...
                 'path',nirsdir);
               
if ~any(ismember(opt.system,{'nirx'}))
  error('Unknown NIRS system %s.',opt.system)
end

% Determine extension for header file
if isdefault.headerExt
  switch(opt.system)
    case 'nirx'
      opt.headerExt = '.hdr';
  end
end
%% Get file(list) and process multiple files
if ischar(file) && ismember('*', file),
  tmpName= get_filelist(file, 'ext',opt.headerExt);
  if isempty(tmpName), error('%s.%s not found', file,opt.headerExt); end
  file= tmpName;
end

% Traverse multiple files
if iscell(file),
  hdr_array= struct([]);
  for ii= 1:length(file),
    hdr= nirsfile_readHeader(file{ii},varargin{:});
    hdr_array= struct_sloppycat(hdr_array, hdr, 'matchsize',1);
  end
  hdr = struct();
  % Create single struct
  fn = fieldnames(hdr_array);
  for ff={fn{:}}
    dum = {hdr_array.(ff{:})};
    if ischar(dum{1}) && numel(unique(dum))==1
      % ** STRING **
      % If all strings are the same, take single string
      dum = unique(dum);
      hdr.(ff{:}) = [dum{:}];
    elseif isscalar(dum{1}) && numel(unique([dum{:}]))==1
      % ** NUMBER **
      % If all numers are the same, take single number
      hdr.(ff{:}) = unique([dum{:}]);
    elseif isvector(dum{1}) && isnumeric(dum{1}) && ...
        isequal(dum{:})
      % ** VECTOR **
      % If all number vectors are the same, take one
      hdr.(ff{:}) = dum{1};
    elseif ~isvector(dum{1}) && isnumeric(dum{1}) && ...
        isequal(dum{:})
      % ** MATRIX **
      % If all number matrices are the same, take one
      hdr.(ff{:}) = dum{1};
    else
      hdr.(ff{:}) = dum;
    end
  end

  return;
end

%% Open and process header
fullName = fullfile(opt.path,[file opt.headerExt]);
fid= fopen(fullName, 'r');
if fid==-1, error(sprintf('%s not found', fullName)); end
[dmy, filename]= fileparts(fullName);

if strcmp(opt.system,'nirx')
  % ****************
  % *** nirx ***
  % ****************
  % General Info
  cs = '[GeneralInfo]'; % current section
  getEntry(fid, cs); 
  hdr.fileName = getEntry(fid, 'FileName=', 0, filename,cs);
  hdr.date = getEntry(fid, 'Date=', 0,[],cs);
  hdr.time = getEntry(fid, 'Time=', 0,[],cs);

  % Imaging Parameters
  cs = '[ImagingParameters]'; % current section
  getEntry(fid, cs); 
  hdr.nSources = getEntry(fid, 'Sources=',1,[],cs); 
  hdr.nDetectors = getEntry(fid, 'Detectors=',1,[],cs); 
  hdr.nWavelengths = getEntry(fid, 'Wavelengths=',1,[],cs); 
  hdr.trigIns = getEntry(fid, 'TrigIns=',0,[],cs); 
  hdr.trigOuts = getEntry(fid, 'TrigOuts=',0,[],cs); 
  hdr.fs= getEntry(fid, 'SamplingRate=',0,[],cs); 

  % Paradigm
  cs = '[Paradigm]';
  getEntry(fid, cs); 
  hdr.stimulusType = getEntry(fid, 'StimulusType=', 0, [],cs);

  % GainSettings
  cs = '[GainSettings]';
  getEntry(fid, cs); 
  % Gain = Sowas wie Impedanzen
  str = getEntry(fid, 'Gains=', 0);
    % Read gain matrix
  if strcmp(str,'#')   
    str= deblank(fgets(fid));
    hdr.gains = [];
    while ~strcmp(str,'#')
      hdr.gains = [hdr.gains; str2num(str)];
      str= deblank(fgets(fid));
    end
  end

  % DataStructure
  cs = '[DataStructure]';
  getEntry(fid, cs); 
  % Save SDKey string as numeric array
  eval(['hdr.SDkey=[' getEntry(fid, 'S-D-Key=',1,[],cs) '];']); 
  str = getEntry(fid, 'S-D-Mask=', 0);
    % Read gain matrix
  if strcmp(str,'#')   
    str= deblank(fgets(fid));
    hdr.SDmask = [];
    while ~strcmp(str,'#')
      hdr.SDmask = [hdr.SDmask; str2num(str)];
      str= deblank(fgets(fid));
    end
  end
  % Wavelength is fixed but not coded in the header
  hdr.wavelengths = [760 850];
end



%% Fertig

fclose(fid);



%% Help functions
function [entry, str]= getEntry(fid, keyword, mandatory, default_value,rewind)
% 'mandatory' - an error is issued if keyword not found
% 'rewind'    - rewind file position back to some pointer (given as string)

if ~exist('mandatory','var'), mandatory=1; end
if ~exist('default_value','var'), default_value=[]; end
if ~exist('rewind','var'), rewind=[]; end
entry= 1;

if keyword(1)=='[', 
  fseek(fid, 0, 'bof');
end
ok= 0;
while ~ok && ~feof(fid),
  str= fgets(fid);
  ok= strncmp(keyword, str, length(keyword));
end
if ~ok,
  if mandatory,
    error(sprintf('keyword <%s> not found', keyword));
  else
    entry= default_value;
    return;
  end
end
if keyword(end)=='=',
  entry= deblank(str(length(keyword)+1:end));
end

% Convert to double if number
if ~isnan(str2double(entry))
  entry = str2double(entry);
end

if ~isempty(rewind)
  getEntry(fid, rewind); 
end
