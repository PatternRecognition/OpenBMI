function [cnt,mrk,mnt,hdr] = nirsfile_loadRaw(file, varargin)
% NIRSFILE_LOADRAW - load NIRS data into BBCI format. By default, preprocessing 
% using the Lambert-Beer transform and the reduction of source-detector
% combinations is performed.
%
% Synopsis:
%   [CNT, MRK, HDR, MNT]= nirsfile_loadRaw(FILE, 'Property1',Value1, ...)
%
% Arguments:
%   FILE: file name (no extension),
%         relative to NIRS_RAW_DIR (if it exists) or otherwise EEG_RAW_DIR 
%         unless the filename is an absolute path
%         FILE may also contain the wildcard symbol '*'. In this case
%         make sure that the order of the files (printed to the terminal)
%         is appropriate.
%         FILE may also be a cell array of file names.
%
% Properties:
%   'path' : directory with raw NIRS data (default NIRS_RAW_DIR (if exists)
%           or EEG_RAW_DIR). If data is file contains an absolute path,
%           path is ignored.
%   'restrict': restrict channels using nirs_restrictMontage (default 1)
%   'source': labels of the sources. If not specified, numbers are used and
%            a warning is issued. The order of the labels has to
%            accord with the physical channels.
%   'detector': labels of the detectors. If not specified, numbers are used and
%            a warning is issued. The order of the labels has to
%            accord with the physical channels.
%   'LBepsilon' : epsilon matrix used in Lambert-Beer transform (default
%                 value should be fine...)
%   'LB':   Whether or not to perform the Lambert-Beer transform (default 1)
%   'LBparam' : specify parameters (key/value pairs) to be transmitted to 
%            nirs_LB as cell array (eg {'opdist' 2})
%   'filt': Filter to be applied to raw data *after* Lambert-Beer transform
%           (if applies).
%           opt.filt must be a struct with fields 'b' and 'a' (as used for
%           the Matlab function filter).
%   'filtType': Sets the type of filtering function used. 1 (default) uses
%           the causal 'proc_filt' function. 2 uses 'proc_filtfilt'.
%   'system': NIRS system used (default 'nirx')
%   'extension' : extension for the data files for wavelengths 1 and 2. If
%           not set, the extension is determined automatically based on the
%           NIRS system.
%   'file' : see getChannelPositions
%
% Parameters to the nirs functions called can be passed to nirsfile_loadRaw 
% and are automatically transmitted.
%
% Remark: 
%   Raw data is in volt (returning photons converted to voltage) and should
%   be converted to absorption values (mmol/l) using the Lambert-Beer transform.
%   If you do not want this, set LB to 0.
%   The two wavelengths (or oxy and deoxy if Lambert-Beer was applied) are
%   stacked behind each other in the x-field as [wl1 wl2] or [oxy deoxy];
%  
%  
% NIRx file structure:
%   .hdr       : header file, contains also the markers in decimal form
%                with both sample-wise and second-wise timestamps
%   .evt       : markers in binary form
%   .wl1 .wl2  : data files for wavelengths 1 and 2, where wl1=short wavelength
%                and wl2=long wavelength
%
% Returns:
%   CNT: struct for continuous signals
%        .fs: sampling interval
%        .x : NIRS signals, either raw voltage counts for the two
%             wavelengths (if LB=0) or oxygenated haemoglobin signals and 
%             de-oxygenated haemoglobin signals. Format: time x (channels*2)
%        .source: source channels
%        .detector: detector channels
%        .clab: channel labels of [source x detector] channels. As a convention,
%               the resulting channel names consist of a concatenation
%               of source and detector, e.g. Pz + Cz -> Pz-Cz.
%        .multiplexing: indicates whether only one source was
%        turned on each time ('single') or whether two sources ('dual')
%        were on simultaneously
%        .signal: indicating the signal type (here 'nirs')
%   MRK: struct of marker information
%   HDR: struct of header information
%
%
% See also: nirs_* nirsfile_*
%
% matthias.treder@tu-berlin.de 2011

if ~isempty(whos('global','NIRS_RAW_DIR'))
  global NIRS_RAW_DIR
  nirsdir = NIRS_RAW_DIR;
else
  global EEG_RAW_DIR
  nirsdir = EEG_RAW_DIR;
end

if ~iscell(file) && isabsolutepath(file)
  nirsdir = [];
end

opt = propertylist2struct(varargin{:});
[opt,isdefault] = ...
    set_defaults(opt, ...
                 'path',nirsdir, ...
                 'clab', [], ...
                 'LB',0,...
                 'LBparam',{}, ...
                 'source',[],'detector',[],...
                 'restrict',1,...
                 'file','5_5', ...
                 'fs', 'raw', ...
                 'filt', [], ...
                 'filtType', 1, ...
                 'system','nirx',...
                 'extension',[], ...
                 'verbose', 0, ...
                 'removeOptodes',0 ...
                );


if ~ismember(opt.system,{'nirx'})
  error('Unknown NIRS system %s.',opt.system)
end

% % For nirx clabs have to be given
% if strcmp(opt.system,'nirx') && (isempty(opt.source) || isempty(opt.detector))
%   error(['For ' opt.system ' you need to specify sources and detectors manually.'],opt.system)
% end

% Define file extensions
if isdefault.extension
  switch(opt.system)
    case 'nirx'
      opt.extension = {'.wl1' '.wl2'};
  end
end

% Convert numeric clab to cell array of strings
if isnumeric(opt.source)
  opt.source = apply_cellwise(num2cell(opt.source),'num2str');
end
if isnumeric(opt.detector)
  opt.detector = apply_cellwise(num2cell(opt.detector),'num2str');
end

%% *** Process multiple files ***

% Has a wildcard -> search folder and convert to cell array of filenames
if ischar(file) && ismember('*', file)
  fp = fileparts(file);
  dd= dir([fullfile(opt.path,file) opt.extension{1}]);
  if isempty(dd)
    error('no files matching %s found', file);
  end
  file = apply_cellwise({dd.name}, inline('x(1:end-4)','x'));
  file = strcat(fp,filesep,file);
end

if iscell(file),
  % Traverse files
  if numel(file)>1
    hdr = nirsfile_readHeader(file);
    T= [];
    fprintf('concatenating files in the following order:\n');
    fprintf('%s\n', vec2str(file));

    for f = file
      [cnt, mrk,dum,mnt]= nirsfile_loadRaw(f{:}, varargin{:});
      T = [T size(cnt.x,1)];
      if strcmp(f{:},file{1})
        ccnt= cnt;
        cmrk= mrk;
      else
        if ~isequal(cnt.clab, ccnt.clab),
          error('source x detector channels are inconsistent across files'); 
        end
        if ~isequal(cnt.fs, ccnt.fs)
          error('inconsistent sampling rate'); 
        end
        ccnt.x= cat(1, ccnt.x, cnt.x);
        mrk.pos= mrk.pos + sum(T(1:end-1));
        cmrk= mrk_mergeMarkers(cmrk, mrk);
      end
    end

    ccnt.T= T;
    if numel(file)>1
      ccnt.title= [ccnt.title ' et al.'];
      ccnt.file= strcat(fileparts(ccnt.file), file);
    end
    cnt = ccnt;
    mrk = cmrk;
    return;
  else
    file = file{1};
  end
end

fullName= fullfile(opt.path,file);

%% **** Read header ****
opt_tmp = copy_struct(opt,'system','path','verbose','headerExt');
hdr=nirsfile_readHeader(file,opt_tmp);
hdr.system = opt.system;

if opt.verbose; fprintf('Source wavelengths: [%s] nm\n',num2str(hdr.wavelengths)); end

%% **** Read marker ****
if nargout>1
  opt_tmp = copy_struct(opt,'system','path','verbose','prefix');
  mrk = nirsfile_readMarker(file,opt_tmp);
  mrk.fs = hdr.fs;
  if opt.verbose; fprintf('Markers read, %d events found.\n',numel(mrk.desc)); end
end

%% **** Read NIRS data ****
cnt.fs= hdr.fs;
cnt.nSources = hdr.nSources;
cnt.nDetectors = hdr.nDetectors;

if strcmp(opt.system,'nirx')
  % Read wavelengths 1 and 2
  wl1 = readDataMatrix([fullName opt.extension{1}]);
  wl2 = readDataMatrix([fullName opt.extension{2}]);

  % Infer from number of columns whether multiplexing was single or dual
  % Source-detector format in single mode
  %   s1-d1,s1-d2,..s1-dN, s2-d1,s2-d2....s2-dN,....
  %   so column 4 is the light going from source 1 to detector 4
  % Source-detector format in dual mode
  % z1-d1,z1-d2,..z1-dN, z2-d1,z2-d2....z2-dN,...
  % where z is a combination of 2 sources. The source-pairs are coupled in
  % a fixed way. For 16 sources, the coupled sources are s1-s9, s2-s10, ...
  % s8-16. So in the raw data column 4 is the light going from s1 and s9 to
  % d4.
  nCol = size(wl1,2);
  if nCol==hdr.nSources * hdr.nDetectors
    cnt.multiplexing = 'single';
  elseif nCol==hdr.nSources * hdr.nDetectors/2
    cnt.multiplexing = 'dual';
    % Only half of data columns in dual mode (because two source were 'on'
    % each time), therefore simply concatenate the data
    wl1 = [wl1 wl1];
    wl2 = [wl2 wl2];
  else
    error 'Number of data columns does not match the number of channels'
  end
  hdr.multiplexing = cnt.multiplexing;
  if opt.verbose
    fprintf(['Expecting %d channels (%d sources, %d detectors) and found '...
      '%d data columns; inferring that multiplexing is ''%s''.\n'], ...
      hdr.nSources*hdr.nDetectors,hdr.nSources,hdr.nDetectors,nCol,cnt.multiplexing)
    if strcmp(cnt.multiplexing,'dual')
      fprintf('Duplicating the %d data columns to assure %d channels.\n',nCol,hdr.nSources*hdr.nDetectors)
    end    
  end  
end

%% Stack wavelengths together in one field
cnt.x = [wl1 wl2];

%% **** Source and detector labels *****
if ~isfield(cnt,'source')
  sourceClab = opt.source;
else
  sourceClab = cnt.source.clab;
end
if ~isfield(cnt,'detector')
  detectorClab = opt.detector;
else
  detectorClab = cnt.detector.clab;
end


%% **** Montage ****
if nargout >= 3 && (isempty(sourceClab) || isempty(detectorClab))  
 % Hadi: made it >= instead of > because of the way calibrate.m calls
  mnt = struct();
  warning('Sources or detectors not found/specified. Making empty montage')
  
elseif nargout >= 3 % Hadi: made it >= instead of > because of the way calibrate.m calls
  opt_tmp = copy_struct(opt,'file','clabPolicy','projection','connector');
  mnt = nirs_getMontage(sourceClab,detectorClab,opt_tmp);
  
  if opt.LB,  app = {' oxy' ' deoxy'};
  else        app = {' wl1' ' wl2'};
  end  
  cnt.clab = cell_flaten({strcat(mnt.clab,app{1}) strcat(mnt.clab,app{2})});
  
  if opt.restrict 
    opt_tmp = copy_struct(opt,'dist','removeOptodes');
    allclab = mnt.clab;
    mnt = nirs_restrictMontage(mnt,opt_tmp); 
    cnt = proc_selectChannels(cnt,mnt.clab,{'ignore' '-'});
  end

end

cnt.wavelengths = hdr.wavelengths;


%% **** Lambert-Beer transform ****
if opt.LB
	cnt = nirs_LB(cnt,opt.LBparam{:});
else
  dat.yUnit = 'V';
end

%% **** Filter ****
if ~isempty(opt.filt),
  switch(opt.filtType)
    case 1
      cnt = proc_filt(cnt,opt.filt.b, opt.filt.a);
    case 2
      cnt = proc_filtfilt(cnt,opt.filt.b, opt.filt.a);
    otherwise
      error('Unknown filt type "%d".\n',opt.filtType)
  end
end

%% *** Return ***
cnt.title= file;
cnt.file= fullName;
cnt.signal = 'nirs';


%% Read data function using 'textscan' (faster than 'textread')
function dat = readDataMatrix(file)
  fid = fopen(file,'r');
  % Read first line and determine nr of columns
  l1 = fgetl(fid);
  nCol = numel(strfind(l1,' ')) + 1;
  % Read
  fseek(fid,0,'bof');
  dat = textscan(fid,repmat('%n',[1 nCol]));
  dat = [dat{:}];
  % Tschuess
  fclose(fid);
end

end