function cnt= readGenericEEG_int(file, chans, fs, from, maxlen)
%cnt=readGenericEEG(file, <chans, fs, from, maxlen>)
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

if file(1)==filesep,
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
else
  chInd= unique(chanind(cnt.clab, chans));
  cnt.clab= {cnt.clab{chInd}};
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
%% this is still a hack
%% allocate memory stepwise to save memory (yes, this is paradox)
step= floor(uChans/4);
cnt.x= int16(zeros(T, step));
cnt.x= cat(2, cnt.x, int16(zeros(T, step)));
cnt.x= cat(2, cnt.x, int16(zeros(T, step)));
cnt.x= cat(2, cnt.x, int16(zeros(T, uChans-size(cnt.x,2))));
%% read data channelwise
for ci= 1:uChans,
  cn= chInd(ci);
  fseek(fid, offset+(cn-1)*cellSize, 'bof');
  cnt.x(:,ci)= int16(fread(fid, [1 T], 'short', cellSize * (lag*nChans-1))');
end    
fclose(fid);
warning('Data is left in int16 format and not properly scaled!');

cnt.title= file;
cnt.file= fullName;
