function [varargout] = eegfile_loadBV(file, varargin)
% EEGFILE_LOADBV - load EEG data which is stored in BrainVision format.
%                  Loading is performed using Matlab code only. EEGFILE_READBV 
%                  should be preferred for performance reasons.
%
% Synopsis:
%   [CNT, MRK, HDR]= eegfile_loadBV(FILE, 'Property1',Value1, ...)
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
%   'start': Start [msec] reading.
%   'maxlen': Maximum length [msec] to be read.
%   'filt': Filter to be applied to raw data before subsampling.
%           opt.filt must be a struct with fields 'b' and 'a' (as used for
%           the Matlab function filter).
%           Note that using opt.filt may slow down loading considerably.
%   'filtType': Sets the type of filtering function used. 1 (default) uses
%           the causal 'proc_filt' function. 2 uses 'proc_filtfilt'.
%   'subsample_fcn': Function that is used for subsampling after filtering, 
%           specified as as string. 'proc_' is automatically prepended.
%           Default 'subsampleByMean'.
%           When not opt.filt is used, data is subsampled by lagging
%           (implemented by reading only every N-th sample, meaning that
%           it works quite fast). 
% Obsolete Properties:
%   'freq': option obsolete, use 'filt' instead.
%   'prec': flag, if true the data are loaded in 16 bit format (BV format) 
%           to save memory storage, an additional output field .scale denotes 
%           the precision of the data, i.e. the real data value is .x*.scale. 
%           Default: false (OPTION is obsolete and will be replaced)
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
                 'filtType', 1, ...
                 'freq', [], ...
                 'linear_derivation', [], ...
                 'outputStructArray',0, ...
                 'verbose', 0);

if ~isempty(opt.ival),
  if ~isdefault.start | ~isdefault.maxlen,
    error('specify either <ival> or <start/maxlen> but not both');
  end
  if ~isempty(opt.ival_sa),
    error('specify either <ival> or <ival_sa> but not both');
  end
  opt.start= opt.ival(1);
  opt.maxlen= diff(opt.ival)+1;
end

if isempty(opt.filt) & ~isequal(opt.subsample_fcn, 'subsampleByLag'),
  warning('When opt.filt is not set subsampling is done by lagging.');
  %% workaround: use opt.filt= struct('b',1, 'a',1);
end

%% prepare return arguments
varargout= cell(1, nargout);

if iscell(file),
  if opt.start~=0 | opt.maxlen~=inf,
    error('concatenation can (so far) only be performed in complete loading');
  end
  [varargout{:}]= eegfile_concatBV(file, varargin{:});
  return;
end

if file(1)==filesep | (~isunix & file(2)==':') 
  fullName= file;
else
  fullName= [EEG_RAW_DIR file];
end

if ischar(fullName) & ismember('*', fullName),
  dd= dir([fullName '.eeg']);
  if isempty(dd),
    error(sprintf('no files matching %s found', [fullName '.eeg']));
  end
  fc= apply_cellwise({dd.name}, inline('x(1:end-4)','x'));
  if length(fc)>1,
    fprintf('concatenating files in the following order:\n');
    fprintf('%s\n', vec2str(fc));
  end
  fullName= strcat(fileparts(fullName), '/', fc);;
  [varargout{:}]= eegfile_loadBV(fullName, opt);
  return;
end

hdr= eegfile_readBVheader(file);
cnt.clab= hdr.clab;
scale= hdr.scale;
raw_fs= hdr.fs;
endian= hdr.endian;

if isequal(opt.fs, 'raw'),
  opt.fs= raw_fs;
end
lag = raw_fs/opt.fs;

if ~isempty(opt.linear_derivation),
  rm_clab= cell_flaten({opt.linear_derivation.rm_clab});
  rmidx= chanind(cnt.clab, rm_clab);
  cnt.clab(rmidx)= [];
  cnt.clab= cat(2, cnt.clab, rm_clab);
end

cnt.fs= opt.fs;
nChans= length(cnt.clab);
if isempty(opt.clab),
  chInd= 1:nChans;
else
  chInd= unique(chanind(cnt.clab, opt.clab));
  cnt.clab= {cnt.clab{chInd}};
end
uChans= length(chInd);

if lag~=round(lag) | lag<1,
  error('fs must be a positive integer divisor of the file''s fs');
end

if ~isempty(opt.freq) & lag>1,
  %% OBSOLETE: this will be removed in future
  if ~isempty(opt.filt),
    error('at most one option may be used: EITHER opt.filt OR opt.freq!');
  end
  freq= 2*[opt.freq]/(lag*opt.fs);
  if max(freq) > 1/lag
    freq = 1/lag;
    order = 5;
  else
    order = 3;
  end
  [opt.filt.b, opt.filt.a]= butter(order, freq);
end

if ~isempty(opt.filt),
  opt_tmp= rmfield(opt_orig, 'filt');       %% this is very unsch�n
  opt_tmp= setfield(opt_tmp, 'fs','raw');
  opt_tmp= setfield(opt_tmp, 'subsample_fcn','subsampleByLag');
  opt_tmp= setfield(opt_tmp, 'verbose',0);
  opt_tmp= setfield(opt_tmp, 'linear_derivation',[]);
  tic;
  for cc= 1:uChans,
    if cc==1 & nargout>1,
      [cnt_sc, mrk]= eegfile_loadBV(file, opt_tmp, 'clab',cnt.clab{cc});
      mrk= mrk_resample(mrk, opt.fs);
    else
      cnt_sc= eegfile_loadBV(file, opt_tmp, 'clab',cnt.clab{cc});
    end
    if opt.filtType == 1,
        cnt_sc= proc_filt(cnt_sc, opt.filt.b, opt.filt.a);
    elseif opt.filtType == 2,
        cnt_sc= proc_filtfilt(cnt_sc, opt.filt.b, opt.filt.a);
    end
    cnt_sc= feval(['proc_' opt.subsample_fcn], cnt_sc, lag);
    if cc==1,
      cnt.x= zeros(size(cnt_sc.x,1), uChans);
      cnt.title= cnt_sc.title;
      cnt.file= cnt_sc.file;
    end
    cnt.x(:,cc)= cnt_sc.x;
    if opt.verbose,
      print_progress(cc, uChans);
    end
  end
  if ~isempty(opt.linear_derivation),
    ld= opt.linear_derivation;
    for cc= 1:length(ld),
      ci= chanind(cnt.clab, ld(cc).chan);
      support= find(ld(cc).filter);
      s2= chanind(cnt.clab, ld(cc).clab(support));
      cnt.x(:,ci)= cnt.x(:,s2) * ld(cc).filter(support);
      cnt.clab{ci}= ld(cc).new_clab;
    end
    cnt= proc_selectChannels(cnt, 'not', rm_clab);
  end
  varargout{1}= cnt;
  if nargout>1
    varargout{2}= mrk;
  end
  return;
end

switch hdr.BinaryFormat
 case 'INT_16',
  cellSize= 2;
  if opt.prec,
    prec= sprintf('%d*short=>short', nChans);
    cnt.scale= scale(chInd);
  else
    prec= sprintf('%d*short', nChans);
  end
  
 case 'DOUBLE',
  if opt.prec,
    error('Refuse to convert double to INT16');
  end
  cellSize= 8;
  prec= sprintf('%d*double', nChans);
  
 case {'IEEE_FLOAT_32','FLOAT_32'},
  if opt.prec,
    error('Refuse to convert double to FLOAT_32');
  end
  cellSize= 4;
  prec= sprintf('%d*float32', nChans);
  
 case {'IEEE_FLOAT_64','FLOAT_64'},
  if opt.prec,
    error('Refuse to convert double to FLOAT_32');
  end
  cellSize= 8;
  prec= sprintf('%d*float64', nChans);
  
 otherwise
  error(sprintf('Precision %s not known.', hdr.BinaryFormat));
end

fid= fopen([fullName '.eeg'], 'r', endian);
if fid==-1, error(sprintf('%s.eeg not found', fullName)); end

fseek(fid, 0, 'eof');
fileLen= ftell(fid);
if ~isempty(opt.ival_sa),
  if ~isdefault.start | ~isdefault.maxlen,
    error('specify either <ival_sa> or <start/maxlen> but not both');
  end
  skip= opt.ival_sa(1)-1;
  nSamples= diff(opt.ival_sa)+1;
else
  skip= max(0, floor(opt.start/1000*raw_fs));
  nSamples= opt.maxlen/1000*raw_fs;
end
nSamples_left_in_file= floor(fileLen/cellSize/nChans) - skip;
nSamples= min(nSamples, nSamples_left_in_file);

if nSamples<0,
  warning('negative number of samples to read');
  [varargout{:}]= deal([]);
end

T= floor(nSamples/lag);
if uChans<nChans,
  opt.channelwise= 1;
end

%% if lagging is needed, pick the sample in the middle
offset= cellSize * (skip + (ceil(lag/2)-1)) * nChans;

%% hack-pt.1 - fix 2GB problem of fread
if exist('verLessThan')~=2 | verLessThan('matlab','7'),
  zgb= 1024*1024*1024*2;
  if offset>=zgb & (opt.channelwise | lag>1),
    postopos= cellSize * lag * nChans;
    overlen= offset - zgb + 1;
    nskips= ceil(overlen/postopos) + 1;  %% '+1' is to be on the save side
    newoffset= offset - nskips*postopos;
    offcut= nskips + 1;
    T= T + nskips;
    offset= newoffset;
  else
    offcut= 1;
  end
end

%% prepare return arguments
varargout= cell(1, nargout);

if ~opt.channelwise,
  
  %% read all channels at once
  fseek(fid, offset, 'bof');
  cnt.x= fread(fid, [nChans T], prec, cellSize * (lag-1)*nChans);
  for ci= 1:uChans,
    cn= chInd(ci);    %% this loopy version is faster than matrix multiplic.
    if ~opt.prec
      cnt.x(ci,:)= scale(cn)*cnt.x(ci,:);
    end
  end
  cnt.x= cnt.x';
else
  
  %% read data channelwise
  %% BB: Who wrote the opt.prec stuff. That looks quite odd.
  if ~ischar(opt.prec)&opt.prec
    cnt.x= repmat(int16(0),[T,uChans]);
    for ci = 1:uChans,
      if opt.verbose,
        fprintf('read channel %3i/%3i   \r', ci, uChans);
      end
      cn= chInd(ci);
      fseek(fid, offset+(cn-1)*cellSize, 'bof');
      cnt.x(:,ci)= [fread(fid, [1 T], '*short', cellSize * (lag*nChans-1))]';
    end
    cnt.scale= scale(chInd);
  else
    cnt.x= zeros(T, uChans);
    for ci= 1:uChans,
      if opt.verbose,
        fprintf('read channel %3i/%3i   \r', ci, uChans);
      end
      cn= chInd(ci);
      fseek(fid, offset+(cn-1)*cellSize, 'bof');
      if ischar(opt.prec)
        % Data precision: float32 or similar. 
        cnt.x(:,ci)= [scale(cn) * ...
            fread(fid, [1 T], opt.prec, cellSize * (lag*nChans-1))]';
      else
        % Data precision: int16
        cnt.x(:,ci)= [scale(cn) * ...
            fread(fid, [1 T], 'short', cellSize * (lag*nChans-1))]';
      end
    end    
  end
  if opt.verbose,
    fprintf('                      \r');
  end
end
fclose(fid);

%% hack-pt.2 - fix 2GB problem of fread
if exist('verLessThan')~=2 | verLessThan('matlab','7'),
  if offcut>1,
    cnt.x= cnt.x(offcut:end,:);
  end
end

cnt.title= file;
cnt.file= fullName;

if ~isempty(opt.linear_derivation),
  ld= opt.linear_derivation;
  for cc= 1:length(ld),
    ci= chanind(cnt.clab, ld(cc).chan);
    support= find(ld(cc).filter);
    s2= chanind(cnt.clab, ld(cc).clab(support));
    cnt.x(:,ci)= cnt.x(:,s2) * ld(cc).filter(support);
    cnt.clab{ci}= ld(cc).new_clab;
  end
  cnt= proc_selectChannels(cnt, 'not', rm_clab);
end

varargout{1}= cnt;
if nargout>1,
  mrk= eegfile_readBVmarkers(file, opt.outputStructArray);
  if ~opt.outputStructArray
    mrkpos_ms= mrk.pos/mrk.fs*1000;
    inival= find(mrkpos_ms>=opt.start & mrkpos_ms<=opt.start+opt.maxlen);
    mrk= mrk_selectEvents(mrk, inival);
    mrk.pos= round( mrk.pos/mrk.fs*cnt.fs - opt.start/1000*cnt.fs );
    mrk.fs= cnt.fs;
  end
  varargout{2}= mrk;
end
if nargout>2,
  varargout{3}= hdr;
end
