function cnt= readGenericEEG(file, chans, fs, from, maxlen,me)
%cnt=readGenericEEG(file, <chans, fs, from, maxlen,me>)
%cnt=readGenericEEG(file, <chans, fs, ival>)
%
% IN   file  - file name (no extension),
%              relative to EEG_RAW_DIR unless beginning with '/' (resp '\')
%      chans - channels to load (labels or indices), [] means all
%      fs    - sampling interval, must be an integer divisor of the
%              sampling interval of the raw data, default: 100
%              fs may also be 'raw' which means sampling rate of raw signals
%      ival  - interval to read, [start end] in msec
%      from  - start [ms] for reading
%      maxlen- maximum length [ms] to read
%      me  - usually subsampling is done by picking, 
%              if me==1, the mean is used for subsampling.
%
% OUT  cnt      struct for contiuous signals
%         .x    - EEG signals (time x channels)
%         .clab - channel labels
%         .fs   - sampling interval
%
% GLOBZ  EEG_RAW_DIR

%% Benjamin Blankertz, FhG-FIRST

global EEG_RAW_DIR

cellSize= 2;  %% 'short' -> 2 bytes

if ~exist('fs', 'var') | isempty(fs), fs=100; end
if ~exist('from', 'var') | isempty(from), from=0; end
if ~exist('maxlen', 'var') | isempty(maxlen),
  if length(from)==2,
    maxlen= diff(from);
    from= from(1);
  else
    maxlen=inf; 
  end
end

if ~exist('me','var') | isempty(me)
  me = 0;
end

if (isunix & file(1)==filesep) | (~isunix & file(2)==':')
  fullName= file;
else
  fullName= [EEG_RAW_DIR file];
end

[cnt.clab, scale, raw_fs, endian]= readGenericHeader(file);
if isequal(fs, 'raw'),
  fs= raw_fs;
end
lag = raw_fs/fs;

if lag~=round(lag) | lag<1,
  error('fs must be a positive integer divisor of the file''s fs');
end
cnt.fs= fs;
nChans= length(cnt.clab);
if ~exist('chans','var') | isempty(chans), 
  chInd= 1:nChans;
elseif iscell(chans)
  chInd= unique(chanind(cnt.clab, chans));
  cnt.clab= {cnt.clab{chInd}};
else
  % chans is a double array
  chInd = unique(chans);
  cnt.clab = {cnt.clab{chInd}};
end

fid= fopen([fullName '.eeg'], 'r', endian);
if fid==-1, error(sprintf('%s.eeg not found', fullName)); end

fseek(fid, 0, 'eof');
fileLen= ftell(fid);
skip= floor(from/1000*raw_fs);
nSamples= floor(fileLen/cellSize/nChans) - skip;
nSamples= min(nSamples, maxlen/1000*raw_fs);

T= floor(nSamples/lag)-1;
uChans= length(chInd);

offset= cellSize * (skip + (ceil(lag/2)-1)) * nChans;
if uChans==nChans & me~=1
%% read all channels at once
  fseek(fid, offset, 'bof');
  prec= sprintf('%d*short', nChans);
  cnt.x= fread(fid, [nChans T], prec, cellSize * (lag-1)*nChans);
  for ci= 1:uChans,
    cn= chInd(ci);    %% this loopy version is faster than matrix multiplic.
    cnt.x(ci,:)= scale(cn)*cnt.x(ci,:);
  end
  cnt.x= cnt.x';
else
%% read data channelwise
  cnt.x= zeros(T, uChans);
  for ci= 1:uChans,
    cn= chInd(ci);
    fseek(fid, offset+(cn-1)*cellSize, 'bof');
    if me==1
      da = fread(fid,[1,nSamples],'short',cellSize*(nChans-1));
      da = da(:,1:lag*T);
      da = reshape(da,[1,lag,T]);
      da = mean(da,2);
      cnt.x(:,ci) = [scale(cn)*permute(da,[1,3,2])]';
    else
      cnt.x(:,ci)= [scale(cn) * ...
          fread(fid, [1 T], 'short', cellSize * (lag*nChans-1))]';
    end
  end    
end
fclose(fid);

cnt.title= file;
cnt.file= fullName;