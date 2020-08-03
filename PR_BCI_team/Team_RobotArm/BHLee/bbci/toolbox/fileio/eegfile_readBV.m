function [varargout] = eegfile_readBV(file, varargin)
% EEGFILE_READBV - load EEG data which is stored in BrainVision format.
%                  C-functions are used for better performance. Use the
%                  slower EEGFILE_LOADBV if this function does not work.
%
% Synopsis:
%   [CNT, MRK, HDR]= eegfile_readBV(FILE, 'Property1',Value1, ...)
%
% Arguments:
%   FILE: file name (no extension),
%         relative to EEG_RAW_DIR unless beginning with '/' (resp '\').
%         FILE may also contain the wildcard symbol '*'. In this case
%         make sure that the order of the files (printed to the terminal)
%         is appropriate.
%         FILE may also be a cell array of file names.
%
% Properties:
%   'clab': Channels to load (labels or indices). Default all
%           (which can be explicitly specified by [])
%   'fs': Sampling interval, must be an integer divisor of the
%         sampling interval of the raw data. fs may also be 'raw' which means 
%         sampling rate of raw signals. Default: 'raw'.
%   'ival': Interval to read, [start end] in msec. It is not checked
%           whether the whole interval could be loaded, or the file is shorter.
%   'ival_sa': Same as 'ival' but [start end] in samples of the downsampled data.
%   'start': Start [msec] reading.
%   'maxlen': Maximum length [msec] to be read.
%   'filt': Filter to be applied to raw data before subsampling.
%           opt.filt must be a struct with fields 'b' and 'a' (as used for
%           the Matlab function filter).
%           Note that using opt.filt may slow down loading considerably.
%   'subsample_fcn': Function that is used for subsampling after filtering, 
%           specified as as string or a vector.
%           Default 'subsampleByMean'. Other 'subsampleByLag'
%           If you specify a vector it has to be the same size as lag.
%
% Remark: 
%   Properties 'ival' and 'start'/'maxlen' are exclusive, i.e., you may only
%   specify one of them.
%
% Returns:
%   CNT: struct for contiuous signals
%        .x: EEG signals (time x channels)
%        .clab: channel labels
%        .fs: sampling interval
%        .scale: if this field is given, the real data are .x*.scale, 
%           and .x is int
%   MRK: struct of marker information
%   HDR: struct of header information
%
% TODO: The function so far can only read specific formats, e.g.,
%       multiplexed, INT_16, ... The function does not even check, whether
%       the file is in this format!
%
% See also: eegfile_*
%
%Hints:
% A low pass filter to get rid of the line noise can be designed as follows:
%  hdr= eegfile_readBVheader(file);
%  Wps= [40 49]/hdr.fs*2;
%  [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
%  [filt.b, filt.a]= cheby2(n, 50, Ws);
% You can also use filtdemo to design your own filters.


% Author(s): blanker@cs.tu-berlin.de

%   2008/06/20/ - Max Sagebaum
%               - refactored eegfile_loadBV to read the data with read_bv.c
%               - removed code fragments marked as obsolete
%               - you can now use the ival option when concatinating
%                 multiple eeg files
%   2008/06/17  - Max Sagebaum
%               - the iir filter was not properly send to read_bv
%   2010/09/09  - Max Sagebaum
%               - There was an bug in the check for the lag




global EEG_RAW_DIR

opt_orig= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt_orig, ...
                 'clab', [], ...
                 'fs', 'raw', ...
                 'start', 0, ...
                 'maxlen', inf, ...
                 'prec',0,...
                 'ival', [], ...
                 'ival_sa', [], ...
                 'subsample_fcn', 'subsampleByMean', ...
                 'channelwise', 0, ...
                 'filt', [], ...
                 'freq', [], ...
                 'linear_derivation', [], ...
                 'outputStructArray',0, ...
                 'marker_format', 'string', ...
                 'targetFormat','bbci',...
                 'verbose', 1);

if ~isempty(opt.ival),
  if ~isdefault.start || ~isdefault.maxlen,
    error('specify either <ival> or <start/maxlen> but not both');
  end
  if ~isempty(opt.ival_sa),
    error('specify either <ival> or <ival_sa> but not both');
  end
  opt.start= opt.ival(1);
  opt.maxlen= diff(opt.ival)+1;
end

%% read the headers an prepare the clab

if ~iscell(file)
  file = {file};
end

fileNames = cell(1,length(file));
% use EEG_RAW_DIR as default dir
for filePos = 1:length(file)
  if file{filePos}(1)==filesep || (~isunix && file{filePos}(2)==':') 
    fileNames{filePos}= file{filePos};
  else
    fileNames{filePos} = [EEG_RAW_DIR file{filePos}];
  end
end

% get all files specified with the file object
fileNamesTemp = {};
for filePos = 1:length(file)

  if ischar(fileNames{filePos}) && ismember('*', fileNames{filePos}),
    dd= dir([fileNames{filePos} '.eeg']);
    if isempty(dd),
      error(sprintf('\nFile not found: %s\n', fileNames{filePos}));
    end
    fc= apply_cellwise({dd.name}, inline('x(1:end-4)','x'));
    
    fileNamesTemp = cat(2,fileNamesTemp,strcat(fileparts(fileNames{filePos}), '/', fc));
  else
    fileNamesTemp = cat(2,fileNamesTemp,{fileNames{filePos}});
  end
end
fileNames = fileNamesTemp;
if length(fileNames)>1,
  if opt.verbose,
    fprintf('concatenating files in the following order:\n');
    fprintf('%s\n', vec2str(fileNames));
  end
end

% now we read all headers and make some consistent checks if neeeded
hdr = cell(1,length(fileNames));
for filePos = 1:length(fileNames)
  
  hdr{filePos} = eegfile_readBVheader(fileNames{filePos}, opt);
  
  % set the clabs and the raw_fs if we are in the loop for the first time
  if(filePos == 1)
    cnt.clab= hdr{filePos}.clab;
    raw_fs= hdr{filePos}.fs;
  end
	
  if ~isequal(cnt.clab, hdr{filePos}.clab),
    warning(['inconsistent clab structure will be repaired ' ...
             'by using the intersection']); 
    cnt.clab = intersect(cnt.clab, hdr{filePos}.clab);
  end
  if isequal(opt.fs, 'raw')
    % if we want to read the raw data check if for each file the raw data
    % is the same
    if~isequal(raw_fs, hdr{filePos}.fs)
      error('inconsistent sampling rate');
    end
  else
    % if we have a specific fs check if for each file we have a positive
    % lag
    lag = hdr{filePos}.fs/opt.fs;
    if lag~=round(lag) || lag<1,
      error('fs must be a positive integer divisor of every file''s fs');
    end
  end
end
clab_in_file= cnt.clab;

% select specified channels
if ~isempty(opt.clab) && strcmp(opt.targetFormat,'bbci'),
  cnt.clab= cnt.clab(chanind(cnt, opt.clab));
end

% sort channels for memory efficient application of linear derivation:
% temporary channels are moved to the end
if ~isempty(opt.linear_derivation),
  rm_clab= cell_flaten({opt.linear_derivation.rm_clab});
  rmidx= chanind(cnt.clab, rm_clab);
  cnt.clab(rmidx)= [];
  cnt.clab= cat(2, cnt.clab, rm_clab);
end

%% prepare the output samples
firstFileToRead = 1;
firstFileSkip = 0;
lastFileToRead = length(fileNames);
lastFileLength=inf;


% check if we want to load the raw data
if isequal(opt.fs, 'raw'),
  opt.fs= raw_fs;
end
cnt.fs= opt.fs;
cnt.title = vec2str(file);
cnt.file = vec2str(fileNames);

nChans= length(cnt.clab);

% get the skip and maxlen values for the data in samples for the new
% sampling rate
if ~isempty(opt.ival_sa),
  if ~isdefault.start || ~isdefault.maxlen,
    error('specify either <ival_sa> or <start/maxlen> but not both');
  end
  skip= opt.ival_sa(1)-1;
  maxlen = diff(opt.ival_sa)+1;
else
  skip= max(0, floor(opt.start/1000*opt.fs));
  maxlen = ceil(opt.maxlen/1000*opt.fs);
end

%get the number of samples for every file and check from which file we have
%to read
nSamples = 0;
dataSamples = 0;
dataSize = zeros(1,length(fileNames));
for filePos = 1:length(fileNames)
  % check if we can read the data with read_bv
  % currently only 16Bit Integers are supported
  switch hdr{filePos}.BinaryFormat
   case 'INT_16',
    cellSize= 2;
    readbv_binformat(filePos)=1;
   case 'INT_32',
    cellSize= 4;
    readbv_binformat(filePos)=2;
   case {'IEEE_FLOAT_32', 'FLOAT_32'},
    cellSize= 4;
    readbv_binformat(filePos)=3;
   case {'IEEE_FLOAT_64', 'FLOAT_64', 'DOUBLE'},
    cellSize= 8;
    readbv_binformat(filePos)=4;
   otherwise
    error('Precision %s not known.', hdr.BinaryFormat);
  end
  
  % open the file to get the size
  fid= fopen([fileNames{filePos} '.eeg'], 'r', hdr{filePos}.endian);
  if fid==-1, error('%s.eeg not found', fileNames{filePos}); end
  fseek(fid, 0, 'eof');
  fileLen= ftell(fid);
  fclose(fid);
  
  curChannels = length(hdr{filePos}.clab);
  curLag = hdr{filePos}.fs/opt.fs;
  samples_in_file = floor(fileLen/(cellSize*curChannels));
  samples_after_subsample = floor(samples_in_file / curLag);
  dataSize(filePos) = samples_after_subsample;
  
  % set the new first file and the first data in this file
  if nSamples <= skip
    firstFileToRead = filePos;
    firstFileSkip = skip - nSamples;
    dataSamples = samples_after_subsample - firstFileSkip;
  else
    dataSamples = dataSamples + samples_after_subsample;
  end
  % advance to the end of the cur file
  nSamples = nSamples + samples_after_subsample;
  
  % if we reach the end set the last file and stop reading
  if nSamples >= (skip + maxlen)
    lastFileToRead = filePos;
    lastFileLength = samples_after_subsample - (nSamples - (skip + maxlen));
    dataSamples = dataSamples - (samples_after_subsample - lastFileLength);
     break;
  else
    % only if we have no maxlen
    lastFileLength = samples_after_subsample;
  end
  
end

%% reading the data
%create the data block for all samples
chosen_clab = cnt.clab;
cnt.x = zeros(dataSamples,nChans);
cnt.T = dataSize;

dataOffset = 0; % the offset for the current file
for filePos = firstFileToRead:lastFileToRead
  % get the channel id for this file
  chanids = chanind(clab_in_file,chosen_clab); % the -1 is for read_bv
  
  read_opt = struct('fs',cnt.fs,'chanidx',chanids);

  if ~isempty(opt.filt)
    read_opt.filt_b = opt.filt.b;
    read_opt.filt_a = opt.filt.a;
  end
  % set the subsample filter 
  lag = hdr{filePos}.fs/opt.fs;
  if ischar(opt.subsample_fcn)
    if strcmp(opt.subsample_fcn,'subsampleByMean')
      read_opt.filt_subsample = ones(1,lag)/lag;
    elseif strcmp(opt.subsample_fct,'subsampleByLag')
      read_opt.filt_subsample = [zeros(1,lag-1) 1];
    else
      error('opt.subsample_fct has to be ''subsampleByMean'' or subsampleByLag');
    end
  else
    read_opt.filt_subsample = opt.subsample_fcn;
  end

  read_hdr = struct('fs',hdr{filePos}.fs,'nChans',hdr{filePos}.NumberOfChannels,...
                    'scale',hdr{filePos}.scale,'endian',hdr{filePos}.endian, ...
                    'BinaryFormat',readbv_binformat(filePos));

  % get the position for the data in the whole data set
  if firstFileToRead == filePos
    firstX = 1;
    firstData = firstFileSkip + 1;
  else
    firstX = lastX + 1;
    firstData = 1;
  end

  if lastFileToRead == filePos
    lastX = nSamples;
    lastData = lastFileLength;
  else
    lastX = firstX + dataSize(filePos) - 1 - firstFileSkip;
    lastData = dataSize(filePos);
  end

  read_opt.data = cnt.x;
  read_opt.dataPos = [firstX lastX firstData lastData] - 1;

  % read the data, read_bv will set the data in cnt.x because of the
  % read_opt.data options
  read_bv([fileNames{filePos} '.eeg'],read_hdr,read_opt);



  %% Markers
  if nargout>1,
    opt_mrk= copy_fields(opt, {'outputStructArray', 'marker_format'});
    curmrk= eegfile_readBVmarkers(fileNames{filePos}, opt_mrk);
    curmrk.time= zeros(1, length(curmrk.pos));
    curmrk= mrk_addIndexedField(curmrk, 'time');
    if ~opt.outputStructArray
      mrk_lag = curmrk.fs/cnt.fs;

      %save the high-resolution timing information in field 'time' in
      %the unit 'msec'.
      curmrk.time= curmrk.pos/curmrk.fs*1000;
      %sample the markers down to the new sample rate and the offset for
      %this file
      curmrk.pos = ceil(curmrk.pos/mrk_lag) + dataOffset;
      % find all markers in the whole intervall
      inival= find(curmrk.pos>=(skip + 1) & curmrk.pos<=(skip + maxlen));
      curmrk= mrk_chooseEvents(curmrk, inival);
      %let the markers start at zero
      curmrk.pos= curmrk.pos - skip;
      curmrk.time= curmrk.time - skip/cnt.fs*1000;
      curmrk.fs= cnt.fs;
    end

    if firstFileToRead == filePos
      mrk = curmrk;
    else
      mrk = mrk_mergeMarkers(mrk, curmrk);
    end
  dataOffset = dataOffset + dataSize(filePos);

  end
  
end
clear read_opt;

if ~isempty(opt.linear_derivation),
  ld= opt.linear_derivation;
  for cc= 1:length(ld),
    ci= chanind(cnt.clab, ld(cc).chan);
    support= find(ld(cc).filter);
    s2= chanind(cnt.clab, ld(cc).clab(support));
    cnt.x(:,ci)= cnt.x(:,s2) * ld(cc).filter(support);
    cnt.clab{ci}= ld(cc).new_clab;
  end
  % delete temporary channels: TODO in a memory efficient way
  idx= chanind(cnt, rm_clab);
  cnt.x(:,idx)= [];
  cnt.clab(idx)= [];
end

varargout= cell(1, nargout);

varargout{1}= cnt;
if nargout > 1,
  varargout{2} = mrk;
end
if nargout>2,
  if(1 == length(hdr))
    varargout{3} = hdr{1};
  else
      varargout{3}= hdr;
  end
end

